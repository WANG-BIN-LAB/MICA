"""
train_and_benchmark.py
=============================================================
MICA Codebase Part II -- Training, hyperparameter selection,
and empirical benchmarking.

Corresponding Methods sections:
  - Design of AE model: training setup
    (Adam, lr 1e-3, batch 64, max 500 epochs, He normal init;
     tenfold lr decay after 10-epoch validation plateau;
     early stopping after 20 static epochs; 70:15:15 split)
  - Optimization and selection of gradient hyperparameters
    (grid: d in {4,5,6,7} x lambda in {0.01,0.05,0.07,0.1})
  - Evaluation metrics (GP / RF / CSC) on empirical data
  - Benchmark methods: PCA, PHATE, DME, unconstrained AE
=============================================================
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

from mica_core import (MICA, build_manifold_reference,
                       build_group_template, coregister_to_template,
                       procrustes_align)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================
# 1. Training / validation / test split (70 : 15 : 15)
# =============================================================
def split_data(fc_list, ref_list, seed=42):
    """Partition subjects into training (70%), validation (15%), test (15%)."""
    idx = np.arange(len(fc_list))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.30, random_state=seed)
    va_idx, te_idx = train_test_split(tmp_idx, test_size=0.50, random_state=seed)
    take = lambda ids, X: [X[i] for i in ids]
    return (take(tr_idx, fc_list), take(va_idx, fc_list), take(te_idx, fc_list),
            take(tr_idx, ref_list), take(va_idx, ref_list),
            take(te_idx, ref_list))


# =============================================================
# 2. MICA training
# =============================================================
def train_mica(fc_train, ref_train, fc_val, ref_val,
               n_regions=360, latent_dim=5, lam=0.05,
               lr=1e-3, batch_size=64, max_epochs=500,
               plateau_patience=10, early_stop_patience=20,
               seed=42, verbose=False):
    """
    Train a MICA model.

    Training protocol (Methods):
      - Adam optimizer, initial learning rate 1e-3, batch size 64,
        maximum 500 epochs, He normal initialization
      - Learning rate decayed tenfold after 10 epochs of validation
        plateau
      - Early stopping after 20 static epochs

    The group-level template is derived exclusively from the training
    references (for UCLA analyses: pass only training NC subjects here).

    Returns
    -------
    model : trained MICA (best validation state restored)
    history : dict with per-epoch train/val losses
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Group template from training references + Procrustes co-registration
    template = build_group_template(ref_train)
    ref_train_al = coregister_to_template(ref_train, template)

    X = torch.tensor(np.array(fc_train), dtype=torch.float32)
    M = torch.tensor(np.array(ref_train_al), dtype=torch.float32)
    Xv = torch.tensor(np.array(fc_val), dtype=torch.float32)
    Mv = torch.tensor(np.array(ref_val), dtype=torch.float32)

    loader = DataLoader(TensorDataset(X, M), batch_size=batch_size,
                        shuffle=True)

    model = MICA(n_regions, latent_dim, lam).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=plateau_patience)

    best_val, best_state, static = np.inf, None, 0
    history = {"train": [], "val": []}

    for epoch in range(max_epochs):
        # ---- training ----
        model.train()
        tr_loss = 0.0
        for fc_b, m_b in loader:
            fc_b, m_b = fc_b.to(DEVICE), m_b.to(DEVICE)
            opt.zero_grad()
            g, fc_hat = model(fc_b)
            loss = model.total_loss(fc_b, fc_hat, g, m_b)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(fc_b)
        tr_loss /= len(X)

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            g_v, fc_v = model(Xv.to(DEVICE))
            va_loss = model.total_loss(Xv.to(DEVICE), fc_v,
                                       Mv.to(DEVICE), Mv.to(DEVICE)).item()
        scheduler.step(va_loss)
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        # ---- early stopping ----
        if va_loss < best_val - 1e-6:
            best_val, static = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            static += 1
            if static >= early_stop_patience:
                if verbose:
                    print(f"  Early stop @ epoch {epoch+1}, "
                          f"best val loss {best_val:.6f}")
                break
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d} | train {tr_loss:.6f} | "
                  f"val {va_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def extract_gradients(model, fc_list, batch_size=128):
    """Extract individual low-dimensional gradients G (V, d)."""
    grads = []
    with torch.no_grad():
        for i in range(0, len(fc_list), batch_size):
            X = torch.tensor(np.array(fc_list[i:i + batch_size]),
                             dtype=torch.float32).to(DEVICE)
            g, _ = model(X)
            grads.append(g.cpu().numpy())
    return np.concatenate(grads, axis=0)


# =============================================================
# 3. Hyperparameter grid search and selection
# =============================================================
def orthogonality_score(G):
    """
    Orthogonality metric: mean absolute Pearson correlation between
    gradient axes (lower = less functional redundancy among axes).
    """
    d = G.shape[1]
    cors = [abs(pearsonr(G[:, i], G[:, j])[0])
            for i in range(d) for j in range(i + 1, d)]
    return float(np.mean(cors))


def grid_search(fc_train, ref_train, fc_val, ref_val,
                d_grid=(4, 5, 6, 7), lam_grid=(0.01, 0.05, 0.07, 0.1),
                seed=42):
    """
    Systematic grid search over d x lambda (16 configurations).
    Latent representations are evaluated on the validation set using:
      - Reconstruction error (fidelity of FC recovery)
      - Orthogonality (functional redundancy among gradient axes)
    The optimal configuration is the point closest to the origin on
    the reconstruction-orthogonality scatter plot, balancing high FC
    reconstruction fidelity and minimal inter-axis redundancy.
    """
    results = []
    for d in d_grid:
        for lam in lam_grid:
            print(f"Training d={d}, lambda={lam} ...")
            model, _ = train_mica(fc_train, ref_train, fc_val, ref_val,
                                  latent_dim=d, lam=lam, seed=seed)
            G_val = extract_gradients(model, fc_val)
            recon = float(np.mean(
                (np.array(fc_val) - reconstruct_fc(G_val)) ** 2))
            ortho = orthogonality_score(G_val)
            results.append({"d": d, "lam": lam, "recon": recon,
                            "ortho": ortho})

    # Min-max normalize both metrics, pick the configuration closest
    # to the origin of the reconstruction-orthogonality plane
    rec = np.array([r["recon"] for r in results])
    ort = np.array([r["ortho"] for r in results])
    rec_z = (rec - rec.min()) / (rec.max() - rec.min() + 1e-12)
    ort_z = (ort - ort.min()) / (ort.max() - ort.min() + 1e-12)
    dist = np.hypot(rec_z, ort_z)
    best = dict(results[int(np.argmin(dist))])
    best["distance"] = float(dist.min())
    print(f"\n>>> Optimal configuration: d={best['d']}, "
          f"lambda={best['lam']} "
          f"(recon={best['recon']:.6f}, ortho={best['ortho']:.4f})")
    return best, results


# =============================================================
# 4. Evaluation metrics (empirical data)
# =============================================================
def reconstruct_fc(G, k_neighbors=10):
    """
    RF evaluation: uniform, non-parametric reverse mapping.
    Each FC row vector is reconstructed as a Gaussian-kernel-weighted
    average of the k nearest rows in the embedding. Cross-check with
    the exact algorithm in Supplementary Methods.
    """
    D = cdist(G, G)
    np.fill_diagonal(D, np.inf)
    fc_hat = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        idx = np.argsort(D[i])[:k_neighbors]
        w = np.exp(-D[i, idx] ** 2)
        fc_hat[i] = (w[:, None] * G[idx]).sum(0) / (w.sum() + 1e-12)
    return fc_hat


def joint_trustworthiness(G, fc, k_neighbors=10):
    """
    GP on empirical data: joint trustworthiness and distance-correlation
    index (see Supplementary Methods for the exact definition).

    Trustworthiness: fraction of k-NN in the embedding that are not
    k-NN in the original space (lower intrusion = higher
    trustworthiness), rescaled to [0, 1].
    """
    V = G.shape[0]
    D_orig = cdist(fc, fc)
    D_emb = cdist(G, G)
    np.fill_diagonal(D_orig, np.inf)
    np.fill_diagonal(D_emb, np.inf)
    nn_orig = np.argsort(D_orig, axis=1)[:, :k_neighbors]
    nn_emb = np.argsort(D_emb, axis=1)[:, :k_neighbors]

    intrusions = 0
    for i in range(V):
        intrusions += len(set(nn_emb[i]) - set(nn_orig[i]))
    trust = 1.0 - (2.0 / (V * k_neighbors * (2 * V - 3 * k_neighbors - 1))) \
        * intrusions

    # Distance correlation: Pearson r between distance matrices
    tri = np.triu_indices(V, k=1)
    dist_corr = pearsonr(D_orig[tri], D_emb[tri])[0]

    return float(0.5 * (trust + dist_corr))


def cross_subject_consistency(grads_list, align_to=None):
    """
    CSC: mean pairwise Pearson correlation between individual subject
    gradients (flattened) within the same scanning session.

    PCA, PHATE, and DME embeddings require post-hoc Procrustes
    alignment prior to CSC calculation (pass `align_to`), whereas AE
    and MICA establish directly aligned latent spaces without
    post-hoc registration (pass `align_to=None`).
    """
    if align_to is not None:
        grads_list = [procrustes_align(g, align_to) for g in grads_list]
    flat = [g.flatten() for g in grads_list]
    n = len(flat)
    cors = [pearsonr(flat[i], flat[j])[0]
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(cors))


# =============================================================
# 5. Benchmark methods
# =============================================================
def run_pca(fc_list, d):
    """PCA on stacked FC row vectors."""
    V = fc_list[0].shape[0]
    X = np.concatenate(fc_list, axis=0)
    emb = PCA(n_components=d).fit_transform(X)
    return [emb[i * V:(i + 1) * V] for i in range(len(fc_list))]


def run_phate(fc_list, d):
    """PHATE on stacked FC row vectors (requires `phate`)."""
    try:
        import phate
    except ImportError:
        print("[WARN] `phate` not installed -- skipping PHATE benchmark.")
        return None
    V = fc_list[0].shape[0]
    X = np.concatenate(fc_list, axis=0)
    emb = phate.PHATE(n_components=d, n_landmark=2000,
                      random_state=42, verbose=False).fit_transform(X)
    return [emb[i * V:(i + 1) * V] for i in range(len(fc_list))]


def run_dme(fc_list, d):
    """Diffusion map embedding, per subject, on a Gaussian-kernel
    affinity derived from |FC|."""
    from sklearn.manifold import SpectralEmbedding
    out = []
    for fc in fc_list:
        aff = np.exp(-np.abs(fc))
        out.append(SpectralEmbedding(
            n_components=d, affinity="precomputed",
            random_state=42).fit_transform(aff))
    return out


def run_ae(fc_list, ref_list, d, n_regions, **kwargs):
    """Unconstrained autoencoder (lambda = 0)."""
    model, _ = train_mica(fc_list, ref_list, fc_list, ref_list,
                          n_regions=n_regions, latent_dim=d,
                          lam=0.0, **kwargs)
    return list(extract_gradients(model, fc_list))


def run_mica(fc_list, ref_list, ftr, rtr, fva, rva, d, n_regions,
             lam=0.05, **kwargs):
    """Full MICA pipeline."""
    model, _ = train_mica(ftr, rtr, fva, rva,
                          n_regions=n_regions, latent_dim=d,
                          lam=lam, **kwargs)
    return list(extract_gradients(model, fc_list))


# =============================================================
# 6. Main workflow (empirical data example)
# =============================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Replace with real data loading, e.g.:
    #   timeseries -> build_fc -> threshold_fc for each subject (Glasser 360)
    # ------------------------------------------------------------------
    print("=" * 66)
    print("MICA empirical training & benchmarking pipeline")
    print("=" * 66)
    print("\nThis script expects a list of sparsified FC matrices "
          "`fc_list` and corresponding manifold references.")
    print("Typical workflow:")
    print("  1. refs, t = build_manifold_reference(fc_list, d=5, k=10)")
    print("  2. split   = split_data(fc_list, refs)          # 70:15:15")
    print("  3. best, results = grid_search(...)             # 16 configs")
    print("  4. model, hist = train_mica(..., d=best['d'], lam=best['lam'])")
    print("  5. G = extract_gradients(model, fc_test)")
    print("  6. evaluate GP / RF / CSC against PCA, PHATE, DME, AE")
