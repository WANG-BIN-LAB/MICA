"""
mica.py
=============================================================
MICA: Manifold-constrained Internal Connectivity Autoencoder
Core module -- diffusion manifold reference construction & model
definition.

Corresponding Methods sections:
  - Construction of manifold reference (Eq. 1-5)
  - Design of AE model (Eq. 6-8)

Dependencies: numpy, scipy, scikit-learn, torch
=============================================================
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS


# =============================================================
# 1. Functional connectivity construction
# =============================================================
def build_fc(timeseries):
    """
    Build a functional connectivity (FC) matrix from regional
    time series via pairwise Pearson correlation.

    Parameters
    ----------
    timeseries : (T, V) ndarray
        Regional mean time series; V = number of cortical parcels.

    Returns
    -------
    FC : (V, V) ndarray
    """
    fc = np.corrcoef(timeseries, rowvar=False)
    fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 0.0)
    return fc


def threshold_fc(fc, top_percent=0.10):
    """
    Retain the top 10% strongest connections per row to
    eliminate noise (Methods: FC sparsification).

    Parameters
    ----------
    fc : (V, V) ndarray
    top_percent : float
        Retention fraction per row (default 0.10).

    Returns
    -------
    FC_thr : (V, V) ndarray
    """
    fc_thr = fc.copy()
    V = fc.shape[0]
    k = int(np.ceil(top_percent * (V - 1)))
    for i in range(V):
        row = fc[i].copy()
        row[i] = -np.inf
        thresh = np.partition(row, -k)[-k]
        fc_thr[i, row < thresh] = 0.0
    return fc_thr


# =============================================================
# 2. Manifold reference construction
# =============================================================
def cosine_distance(fc, eps=1e-12):
    """
    Eq. (1): Inter-regional cosine distance
        C_ab = 1 - (FC_a . FC_b) / (||FC_a||_2 * ||FC_b||_2)

    Smaller C_ab indicates more similar FC between regions a and b.
    """
    norms = np.linalg.norm(fc, axis=1, ord=2, keepdims=True)
    norms = np.clip(norms, eps, None)
    fc_norm = fc / norms
    C = 1.0 - fc_norm @ fc_norm.T
    return np.clip(C, 0.0, None)


def knn_adjacency(C, k=10):
    """
    For each region a, identify its nearest neighbors as the regions
    with the k smallest entries in the a-th row of C (self-loop
    excluded). Non-neighbor entries are set to zero.
    """
    A = np.zeros_like(C)
    for i in range(C.shape[0]):
        row = C[i].copy()
        row[i] = np.inf
        idx = np.argsort(row)[:k]
        A[i, idx] = C[i, idx]
    return A


def heat_kernel(C_knn, k=10):
    """
    Eq. (2): Heat-kernel edge weights
        K_ab = exp( -C_ab^2 / (2 * sigma^2) )
    where sigma = mean distance to the k-th nearest neighbor
    across all nodes.
    """
    V = C_knn.shape[0]
    kth_dists = []
    for i in range(V):
        nz = C_knn[i][C_knn[i] > 0]
        if len(nz) >= k:
            kth_dists.append(np.sort(nz)[k - 1])
    sigma = np.mean(kth_dists)
    K = np.exp(-C_knn ** 2 / (2.0 * sigma ** 2))
    K[C_knn == 0] = 0.0
    return K


def reciprocal_reward(K):
    """
    Eq. (3): Reciprocal reward scheme to yield W
        W_ab = K_ab * ( 1 + min(K_ab, K_ba) * (1 - |K_ab - K_ba|) )

    Strong, balanced reciprocal connectivity yields a coefficient > 1,
    whereas unidirectional or highly asymmetric connections remain
    close to 1 (accounting for directional graph asymmetry).
    """
    K_min = np.minimum(K, K.T)
    K_diff = np.abs(K - K.T)
    return K * (1.0 + K_min * (1.0 - K_diff))


def transition_matrix(W, eps=1e-12):
    """
    Eq. (4): Row-normalize W to obtain the transition probability
    matrix P
        P_ab = W_ab / sum_c W_ac
    """
    row_sum = W.sum(axis=1, keepdims=True)
    return W / np.clip(row_sum, eps, None)


def von_neumann_entropy(P, eps=1e-12):
    """
    Von Neumann entropy of a diffusion matrix, used to determine
    the diffusion time t via its knee point.
    """
    eigvals = np.clip(np.linalg.eigvalsh(P), eps, None)
    p_norm = eigvals / eigvals.sum()
    return -np.sum(p_norm * np.log(p_norm))


def find_knee_point(curve):
    """
    Locate the knee point of a curve by maximizing the perpendicular
    distance to the line joining the first and last points.

    Returns
    -------
    t_knee : int (1-based step)
    """
    x = np.arange(1, len(curve) + 1, dtype=float)
    y = curve
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    d = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    d /= np.hypot(y1 - y0, x1 - x0)
    return int(x[np.argmax(d)])


def diffuse(P, t=None, t_max=30):
    """
    Compute the multi-step diffusion matrix P^t. The diffusion time t
    is determined via the knee point of the Von Neumann entropy.

    Returns
    -------
    Pt : (V, V) ndarray
    t_used : int
    """
    Pt = P.copy()
    Pt_list = [Pt.copy()]
    curve = []
    for _ in range(2, t_max + 1):
        Pt = Pt @ P
        Pt_list.append(Pt.copy())
        curve.append(von_neumann_entropy(Pt))
    if t is None:
        t = find_knee_point(np.array(curve))
    return Pt_list[t - 1], t


def potential_distance(Pt, eps=1e-12):
    """
    Eq. (5): Potential distance between regions
        D_ab = || log(P^t_a) - log(P^t_b) ||_2
    where P^t_a is the a-th row of P^t.
    """
    logP = np.log(np.clip(Pt, eps, None))
    return cdist(logP, logP, metric="euclidean")


def mds_embed(D, d, random_state=42):
    """
    Multidimensional Scaling (MDS) reduces D to d dimensions,
    generating a reference matrix M (V, d).
    """
    mds = MDS(n_components=d, dissimilarity="precomputed",
              random_state=random_state, normalized_stress="auto")
    return mds.fit_transform(D)


def procrustes_align(source, target, scaling=False):
    """
    Procrustes alignment (rotation + translation, optional scaling)
    of `source` onto `target`. Reflections are disallowed.
    Used to co-register individual reference manifolds to a
    group-level template.
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
        ns, nt = np.linalg.norm(src_c), np.linalg.norm(tgt_c)
        if ns > 1e-12:
            aligned *= (nt / ns)
    return aligned


def build_manifold_reference(fc_list, d=5, k=10, t=None, t_max=30):
    """
    Full manifold reference construction pipeline.

    Parameters
    ----------
    fc_list : list of (V, V) ndarray
        Individual (sparsified) FC matrices.
    d : int
        Latent dimensionality.
    k : int
        Number of nearest neighbors (default 10).
    t : int or None
        Diffusion time; None = determined via Von Neumann entropy knee.
    t_max : int
        Maximum diffusion steps for the entropy curve.

    Returns
    -------
    ref_manifolds : list of (V, d) ndarray
    t_used : int
    """
    ref_manifolds, t_used = [], None
    for fc in fc_list:
        C = cosine_distance(fc)
        C_knn = knn_adjacency(C, k=k)
        K = heat_kernel(C_knn, k=k)
        W = reciprocal_reward(K)
        P = transition_matrix(W)
        Pt, t_used = diffuse(P, t=t, t_max=t_max)
        D = potential_distance(Pt)
        ref_manifolds.append(mds_embed(D, d=d))
    return ref_manifolds, t_used


def build_group_template(ref_manifolds):
    """
    Group-level template = mean of individual reference manifolds.
    NOTE: For clinical (UCLA) analyses, derive the template
    exclusively from training NC subjects to prevent patient data
    from influencing the coordinate framework.
    """
    return np.mean(ref_manifolds, axis=0)


def coregister_to_template(ref_manifolds, template):
    """Procrustes-align all individual reference manifolds to the group template."""
    return [procrustes_align(M, template) for M in ref_manifolds]


# =============================================================
# 3. MICA autoencoder (Design of AE model)
# =============================================================
class MICA(nn.Module):
    """
    Manifold-constrained autoencoder.

    Encoder: 360 -> 256 -> 128 -> 64 -> d  (ReLU, He normal init)
    Decoder: d -> 64 -> 128 -> 256 -> 360  (symmetric)

    Composite objective (Eq. 6-8):
        L_total = Lr(FC, FC_hat) + lambda * Lm(G, M)
    where Lr is the reconstruction error and Lm the manifold
    regularization term; lambda >= 0 controls regularization strength.
    """

    def __init__(self, n_regions=360, latent_dim=5, lam=0.05):
        super().__init__()
        self.n_regions = n_regions
        self.latent_dim = latent_dim
        self.lam = lam

        self.encoder = nn.Sequential(
            nn.Linear(n_regions, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, n_regions),
        )
        self._init_weights()

    def _init_weights(self):
        """He normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        g = self.encoder(x)        # (B, d)  low-dimensional functional gradients
        fc_hat = self.decoder(g)   # (B, V)  reconstructed FC row vectors
        return g, fc_hat

    @staticmethod
    def reconstruction_loss(fc, fc_hat):
        """
        Eq. (7): Lr = (1/V^2) * sum_a sum_b (FC_ab - FC_hat_ab)^2
        """
        return torch.mean((fc - fc_hat) ** 2)

    @staticmethod
    def manifold_loss(G, M):
        """
        Eq. (8): Lm = (1/(V*d)) * sum_a sum_k (G_ak - M_ak)^2
        """
        return torch.mean((G - M) ** 2)

    def total_loss(self, fc, fc_hat, G, M):
        """Eq. (6)."""
        return (self.reconstruction_loss(fc, fc_hat)
                + self.lam * self.manifold_loss(G, M))


# =============================================================
# 4. Self-test
# =============================================================
if __name__ == "__main__":
    V, N = 360, 20
    rng = np.random.default_rng(0)
    latent_true = rng.normal(size=(N, V, 5))
    fc_list = [threshold_fc(np.tanh(lat @ lat.T / V)) for lat in latent_true]

    refs, t = build_manifold_reference(fc_list, d=5, k=10)
    template = build_group_template(refs)
    aligned = coregister_to_template(refs, template)
    print(f"[OK] Diffusion time t = {t}, reference shape: {aligned[0].shape}")

    model = MICA(n_regions=V, latent_dim=5, lam=0.05)
    x = torch.tensor(fc_list[0], dtype=torch.float32)
    m = torch.tensor(aligned[0], dtype=torch.float32)
    g, fc_hat = model(x)
    print(f"[OK] MICA forward pass, total loss = "
          f"{model.total_loss(x, fc_hat, g, m).item():.6f}")
