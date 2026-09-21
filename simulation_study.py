"""
simulation_study.py
=============================================================
MICA Codebase —— Simulation Study
=============================================================
Benchmarking recovery of known low-dimensional manifolds.

Simulation design (Methods: "Simulation study"):
  - 200 synthetic subjects
  - Predefined structural architecture: 3 gradients across 200 nodes
  - Two main modules, each containing two submodules
  - Subject-specific noise:
      * Sparse uniform perturbations (+/-0.2 with probability 0.1)
      * Node-level Gaussian noise (SD = 0.1 across 20% of nodes per module)
  - Perturbed gradients rotated by a shared random orthogonal matrix Q
  - Synthetic FC derived from node-wise Pearson correlation profiles

This design preserves a fixed global manifold topology across subjects
while introducing realistic individual variability.

All five benchmarked methods are evaluated:
  PCA, PHATE, Diffusion Map Embedding (DME), unconstrained AE, MICA

Evaluation metrics (Methods: "Evaluation metrics"):
  - GP  (Geometric preservation): Pearson correlation between
        ground-truth gradients and Procrustes-aligned recovered embeddings
  - RF  (Reconstruction fidelity): FC reconstructed from low-dimensional
        embeddings via a uniform, non-parametric reverse-mapping algorithm
  - CSC (Cross-subject consistency): mean pairwise Pearson correlation
        between individual subject gradients within the same session.
        PCA / PHATE / DME require post-hoc Procrustes alignment prior to
        CSC calculation, whereas AE and MICA yield directly aligned
        latent spaces without post-hoc registration.

Dependencies: numpy, scipy, scikit-learn, torch, (optional) phate
=============================================================
"""

import numpy as np
import torch
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from sklearn.manifold import SpectralEmbedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================
# 1. Synthetic data generation
# =============================================================
def simulate_subjects(n_subjects=200, n_nodes=200, n_gradients=3,
                      n_modules=2, n_submodules=2,
                      perturb_prob=0.1, perturb_amp=0.2,
                      noise_sd=0.1, noise_frac=0.2, seed=42):
    """
    Generate the synthetic cohort.

    Parameters
    ----------
    n_subjects : int
        Number of synthetic subjects (default 200).
    n_nodes : int
        Number of cortical nodes (default 200).
    n_gradients : int
        Number of ground-truth gradient axes (default 3).
    n_modules : int
        Number of main modules (default 2).
    n_submodules : int
        Number of submodules per main module (default 2).
    perturb_prob : float
        Probability of sparse uniform perturbation per entry (default 0.1).
    perturb_amp : float
        Amplitude of uniform perturbations, applied as +/-amp (default 0.2).
    noise_sd : float
        Standard deviation of node-level Gaussian noise (default 0.1).
    noise_frac : float
        Fraction of nodes per module receiving Gaussian noise (default 0.2).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    fc_list : list of (n_nodes, n_nodes) ndarray
        Synthetic FC matrices (node-wise Pearson correlation profiles).
    true_grads : list of (n_nodes, n_gradients) ndarray
        Ground-truth (perturbed, rotated) gradients per subject.
    base_gradient : (n_nodes, n_gradients) ndarray
        The shared, noise-free rotated base manifold.
    """
    rng = np.random.default_rng(seed)
    nodes_per_submodule = n_nodes // (n_modules * n_submodules)
    nodes_per_module = n_nodes // n_modules

    # --- 1.1 Hierarchical base architecture -------------------------
    # Each gradient axis assigns systematically offset values to
    # (module, submodule) blocks, producing a fixed multi-scale
    # organization: two main modules, each with two submodules.
    base = np.zeros((n_nodes, n_gradients))
    for g in range(n_gradients):
        for mod in range(n_modules):
            for sub in range(n_submodules):
                lo = (mod * n_submodules + sub) * nodes_per_submodule
                base[lo:lo + nodes_per_submodule, g] = (
                    mod * 1.5 + sub * 0.5 + g * 0.8 + rng.normal(0, 0.05)
                )

    # --- 1.2 Shared random orthogonal rotation ----------------------
    # A single rotation Q is shared across all subjects, so the global
    # manifold topology is preserved; only subject-specific noise varies.
    Q, _ = np.linalg.qr(rng.normal(size=(n_gradients, n_gradients)))
    base_rotated = base @ Q

    # --- 1.3 Subject-specific perturbations --------------------------
    fc_list, true_grads = [], []
    for _ in range(n_subjects):
        g = base_rotated.copy()

        # (a) Sparse uniform perturbations: +/-0.2 with probability 0.1
        mask = rng.random(g.shape) < perturb_prob
        g[mask] += rng.choice([-perturb_amp, perturb_amp],
                              size=int(mask.sum()))

        # (b) Node-level Gaussian noise: SD = 0.1 across 20% of the
        #     nodes within each main module
        for mod in range(n_modules):
            module_nodes = np.arange(mod * nodes_per_module,
                                     (mod + 1) * nodes_per_module)
            noisy_nodes = rng.choice(module_nodes,
                                     size=int(noise_frac * nodes_per_module),
                                     replace=False)
            g[noisy_nodes] += rng.normal(0, noise_sd,
                                         size=(len(noisy_nodes), n_gradients))

        # (c) Synthetic FC from node-wise Pearson correlation profiles
        fc = np.corrcoef(g, rowvar=False)
        fc = np.nan_to_num(fc, nan=0.0)
        np.fill_diagonal(fc, 0.0)

        fc_list.append(fc)
        true_grads.append(g)

    return fc_list, true_grads, base_rotated


# =============================================================
# 2. Benchmark methods
# =============================================================
def run_pca(fc_list, d):
    """
    Principal Component Analysis applied to stacked FC row vectors.
    Returns a list of (n_nodes, d) embeddings, one per subject.
    """
    n_nodes = fc_list[0].shape[0]
    X = np.concatenate(fc_list, axis=0)               # (N * V, V)
    emb = PCA(n_components=d).fit_transform(X)        # (N * V, d)
    return [emb[i * n_nodes:(i + 1) * n_nodes] for i in range(len(fc_list))]


def run_phate(fc_list, d):
    """
    PHATE embedding of stacked FC row vectors.
    Requires the `phate` package; returns None if unavailable.
    """
    try:
        import phate
    except ImportError:
        print("[WARN] `phate` not installed -- skipping PHATE benchmark.")
        return None
    n_nodes = fc_list[0].shape[0]
    X = np.concatenate(fc_list, axis=0)
    emb = phate.PHATE(n_components=d, n_landmark=2000,
                      random_state=42, verbose=False).fit_transform(X)
    return [emb[i * n_nodes:(i + 1) * n_nodes] for i in range(len(fc_list))]


def run_dme(fc_list, d):
    """
    Diffusion Map Embedding, computed per subject on a Gaussian-kernel
    affinity matrix derived from |FC|.
    """
    embeddings = []
    for fc in fc_list:
        affinity = np.exp(-np.abs(fc))
        emb = SpectralEmbedding(n_components=d, affinity="precomputed",
                                random_state=42).fit_transform(affinity)
        embeddings.append(emb)
    return embeddings


def run_unconstrained_ae(fc_list, d, n_regions, lam=0.0, **train_kwargs):
    """
    Unconstrained autoencoder: identical architecture to MICA but with
    manifold regularization weight lambda = 0.
    `train_mica` and `MICA` are imported from the MICA codebase.
    """
    from mica_model import MICA, train_mica, extract_gradients  # your module
    dummy_ref = [np.zeros_like(fc) for fc in fc_list]
    model, _ = train_mica(fc_list, dummy_ref, fc_list, dummy_ref,
                          n_regions=n_regions, latent_dim=d, lam=lam,
                          **train_kwargs)
    return list(extract_gradients(model, fc_list))


def run_mica(fc_list, d, n_regions, lam=0.05, k=10, **train_kwargs):
    """
    Full MICA pipeline: diffusion-manifold reference construction
    followed by manifold-constrained autoencoder training.
    """
    from mica_model import (build_manifold_reference, train_mica,
                            extract_gradients)
    refs, _ = build_manifold_reference(fc_list, d=d, k=k)
    model, _ = train_mica(fc_list, refs, fc_list, refs,
                          n_regions=n_regions, latent_dim=d, lam=lam,
                          **train_kwargs)
    return list(extract_gradients(model, fc_list))


# =============================================================
# 3. Evaluation metrics (GP / RF / CSC)
# =============================================================
def procrustes_align(source, target, scaling=False):
    """
    Ordinary Procrustes alignment (rotation, optional scaling, translation)
    of `source` onto `target`. Reflections are disallowed.
    """
    src_c = source - source.mean(0, keepdims=True)
    tgt_c = target - target.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(src_c.T @ tgt_c)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    aligned = src_c @ R
    if scaling:
        norm_src = np.linalg.norm(src_c)
        norm_tgt = np.linalg.norm(tgt_c)
        if norm_src > 1e-12:
            aligned *= (norm_tgt / norm_src)
    return aligned


def geometric_preservation(G_rec, G_true):
    """
    GP (simulated data): mean |Pearson r| between ground-truth gradients
    and Procrustes-aligned recovered embeddings, per axis.

    Parameters
    ----------
    G_rec  : (V, d_gt) recovered embedding
    G_true : (V, d_gt) ground-truth gradients

    Returns
    -------
    float in [0, 1]; higher is better.
    """
    d = G_true.shape[1]
    G_al = procrustes_align(G_rec[:, :d], G_true)
    cors = [abs(pearsonr(G_al[:, i], G_true[:, i])[0]) for i in range(d)]
    return float(np.mean(cors))


def reverse_map_fc(G, k_neighbors=10):
    """
    RF: uniform, non-parametric reverse mapping.
    Each FC row vector is reconstructed as a Gaussian-kernel-weighted
    average of the k nearest rows in the low-dimensional embedding.
    Replace with the exact algorithm from the Supplementary Methods
    if it differs.

    Parameters
    ----------
    G : (V, d) low-dimensional embedding of FC row vectors

    Returns
    -------
    fc_hat : (V, V) reconstructed FC matrix
    """
    D = cdist(G, G)
    np.fill_diagonal(D, np.inf)
    fc_hat = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        idx = np.argsort(D[i])[:k_neighbors]
        w = np.exp(-D[i, idx] ** 2)
        fc_hat[i] = (w[:, None] * G[idx]).sum(0) / (w.sum() + 1e-12)
    return fc_hat


def reconstruction_fidelity(fc_list, embeddings):
    """
    RF: mean squared error between original and reverse-mapped FC,
    averaged over subjects. Lower is better.

    Parameters
    ----------
    fc_list   : list of (V, V) original FC matrices
    embeddings: list of (V, d) low-dimensional embeddings
    """
    errors = []
    for fc, G in zip(fc_list, embeddings):
        fc_hat = reverse_map_fc(G)
        errors.append(np.mean((fc - fc_hat) ** 2))
    return float(np.mean(errors))


def cross_subject_consistency(embeddings, align_to=None):
    """
    CSC: mean pairwise Pearson correlation between flattened individual
    gradient matrices within the same scanning session.

    Parameters
    ----------
    embeddings : list of (V, d) per-subject gradients.
    align_to   : optional reference (V, d). If provided, every embedding
                 is first Procrustes-aligned to it (required for PCA,
                 PHATE, and DME, whose sign/rotation is arbitrary).
                 AE and MICA produce directly aligned latent spaces and
                 need no post-hoc registration (pass align_to=None).
    """
    if align_to is not None:
        embeddings = [procrustes_align(e, align_to) for e in embeddings]
    flat = [e.flatten() for e in embeddings]
    n = len(flat)
    cors = [pearsonr(flat[i], flat[j])[0]
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(cors))


# =============================================================
# 4. Main experiment
# =============================================================
def main():
    N_SUBJECTS = 200
    N_NODES = 200
    N_GRADIENTS = 3          # ground-truth manifold dimensionality
    D_EMBED = 3              # embedding dimensionality for all methods
    SEED = 42

    print("=" * 66)
    print("MICA Simulation Study")
    print("=" * 66)

    # -------------------------------------------------------------
    # Step 1. Generate synthetic cohort
    # -------------------------------------------------------------
    print(f"\n[1] Generating {N_SUBJECTS} synthetic subjects "
          f"({N_NODES} nodes, {N_GRADIENTS} gradients, "
          f"2 modules x 2 submodules)...")
    fc_list, true_grads, base = simulate_subjects(
        n_subjects=N_SUBJECTS, n_nodes=N_NODES, n_gradients=N_GRADIENTS,
        seed=SEED)
    print(f"    FC shape: {fc_list[0].shape}; "
          f"ground-truth manifold: {base.shape}")

    # -------------------------------------------------------------
    # Step 2. Run all five benchmarked methods
    # -------------------------------------------------------------
    print("\n[2] Running benchmark methods...")
    methods = {}
    methods["PCA"] = run_pca(fc_list, D_EMBED)
    methods["PHATE"] = run_phate(fc_list, D_EMBED)
    methods["DME"] = run_dme(fc_list, D_EMBED)

    try:
        from mica_model import train_mica, extract_gradients
        has_mica = True
        methods["AE"] = run_unconstrained_ae(
            fc_list, D_EMBED, n_regions=N_NODES, seed=SEED)
        methods["MICA"] = run_mica(
            fc_list, D_EMBED, n_regions=N_NODES, lam=0.05, seed=SEED)
    except ImportError:
        has_mica = False
        print("[WARN] `mica_model` module not found -- "
              "skipping AE and MICA benchmarks.")

    # -------------------------------------------------------------
    # Step 3. Evaluate GP / RF / CSC
    # -------------------------------------------------------------
    print("\n[3] Evaluation on simulated data")
    header = f"{'Method':8s} | {'GP':>8s} | {'RF (MSE)':>10s} | {'CSC':>8s}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for name, embs in methods.items():
        if embs is None:
            continue

        # GP: mean over subjects of |r| between aligned embedding and truth
        gp = np.mean([geometric_preservation(embs[i], true_grads[i])
                      for i in range(N_SUBJECTS)])

        # RF: reverse-mapping reconstruction error
        rf = reconstruction_fidelity(fc_list, embs)

        # CSC: PCA / PHATE / DME need post-hoc Procrustes alignment
        #      (aligned to the first subject as reference);
        #      AE / MICA are directly aligned, no registration needed.
        if name in ("PCA", "PHATE", "DME"):
            csc = cross_subject_consistency(embs, align_to=embs[0])
        else:
            csc = cross_subject_consistency(embs, align_to=None)

        print(f"{name:8s} | {gp:8.4f} | {rf:10.6f} | {csc:8.4f}")

    print("-" * len(header))
    print("\nDone. Higher GP / CSC and lower RF indicate better performance.")


if __name__ == "__main__":
    main()
