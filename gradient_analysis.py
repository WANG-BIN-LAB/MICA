"""
gradient_analysis.py
=============================================================
MICA Codebase Part III -- Gradient analysis.

Corresponding Methods sections:
  - Gradient metrics (Eq. 9-13):
      * Global eccentricity        E_m,k      (Eq. 11)
      * Inter-module distance      Dinter     (Eq. 12)
      * Intra-module eccentricity  Dintra_m,k (Eq. 13)
  - Model stability:
      * Test-retest reliability via single-score, absolute-agreement
        ICC (two-way random-effects model)
      * Cross-atlas similarity (CAS) and cross-dataset similarity
        (CDS) via vertex-wise Pearson correlations on fsLR-32k
=============================================================
"""

import numpy as np
from scipy.stats import pearsonr


# =============================================================
# 1. Gradient spatial metrics (Gradient metrics)
# =============================================================
def _network_centroid(G, members):
    """Eq. (9): Centroid of network m along all axes, C_m,k."""
    return np.mean(G[np.asarray(members)], axis=0)


def _global_centroid(G):
    """Eq. (10): Global centroid across all N cortical regions."""
    return np.mean(G, axis=0)


def global_eccentricity(G, networks):
    """
    Eq. (11): Global eccentricity per network and axis
        E_m,k = (C_m,k - C_global,k)^2

    Quantifies how far a network is displaced from the brain-wide
    average along a given axis. A larger E_m,k indicates that network
    m occupies an extreme position along that axis.

    Returns
    -------
    {network_name: (d,) ndarray of per-axis eccentricities}
    """
    c_global = _global_centroid(G)
    return {name: (_network_centroid(G, m) - c_global) ** 2
            for name, m in networks.items()}


def inter_module_distance(G, networks):
    """
    Eq. (12): Inter-module distance for all unique network pairs
        Dinter_m,n,k = (C_m,k - C_n,k)^2

    Elevated values indicate that networks m and n are segregated
    along that axis.

    Returns
    -------
    {(net_a, net_b): (d,) ndarray of per-axis distances}
    """
    centroids = {name: _network_centroid(G, m)
                 for name, m in networks.items()}
    names = list(networks.keys())
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            out[(a, b)] = (centroids[a] - centroids[b]) ** 2
    return out


def intra_module_eccentricity(G, networks):
    """
    Eq. (13): Intra-module eccentricity per network and axis
        Dintra_m,k = sum_{i in m} (g_i,k - C_m,k)^2

    Reduced values signify that regions within network m are tightly
    aggregated along that axis; higher values indicate internal
    functional differentiation.

    Returns
    -------
    {network_name: (d,) ndarray of per-axis dispersions}
    """
    out = {}
    for name, members in networks.items():
        members = np.asarray(members)
        c_m = _network_centroid(G, members)
        out[name] = ((G[members] - c_m) ** 2).sum(axis=0)
    return out


def summarize_axis_metrics(G, networks, summary_networks=None):
    """
    Aggregate module-level metrics into global summary metrics per
    gradient axis:
      - Global eccentricity & intra-module eccentricity:
            averaged across the seven Yeo networks
      - Inter-module distance:
            averaged across all unique network pairs

    Returns
    -------
    summary : dict {metric_name: (d,) ndarray}
    """
    nets = summary_networks if summary_networks is not None \
        else list(networks.keys())

    E = global_eccentricity(G, networks)
    Dinter = inter_module_distance(G, networks)
    Dintra = intra_module_eccentricity(G, networks)

    return {
        "global_eccentricity":
            np.mean([E[n] for n in nets if n in E], axis=0),
        "intra_module_eccentricity":
            np.mean([Dintra[n] for n in nets if n in Dintra], axis=0),
        "inter_module_distance":
            np.mean([v for (a, b), v in Dinter.items()
                     if a in nets and b in nets], axis=0),
    }


# =============================================================
# 2. Test-retest reliability: ICC(2,1)
# =============================================================
def icc_single_absolute(data, eps=1e-12):
    """
    Single-score, absolute-agreement, two-way random-effects ICC(2,1).

    Parameters
    ----------
    data : (n_targets, n_raters) ndarray
        E.g., (n_subjects, n_sessions) values for one gradient axis.

    Returns
    -------
    ICC(2,1) : float
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    grand = data.mean()
    MSR = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    MSC = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = (data - data.mean(axis=1, keepdims=True)
             - data.mean(axis=0, keepdims=True) + grand)
    SSE = (resid ** 2).sum()
    MSE = SSE / ((n - 1) * (k - 1))
    return float((MSR - MSE) /
                 (MSR + (k - 1) * MSE + k * (MSC - MSE) / n + eps))


def test_retest_reliability(grads_session1, grads_session2):
    """
    Test-retest reliability: per-axis ICC(2,1) between two scanning
    sessions.

    Parameters
    ----------
    grads_session1/2 : list of (V, d) per-subject gradients.

    Returns
    -------
    iccs : (d,) ndarray of per-axis ICC values
    """
    s1 = np.stack(grads_session1)   # (N, V, d)
    s2 = np.stack(grads_session2)
    d = s1.shape[2]
    iccs = []
    for axis in range(d):
        mat = np.stack([s1[:, :, axis], s2[:, :, axis]], axis=1)
        iccs.append(icc_single_absolute(mat))
    return np.array(iccs)


# =============================================================
# 3. Cross-atlas / cross-dataset similarity (CAS / CDS)
# =============================================================
def parcels_to_surface(parcel_values, vertex_labels, n_vertices=32768):
    """
    Project region-level gradients onto the fsLR-32k surface via
    nearest-neighbor label expansion.

    Parameters
    ----------
    parcel_values : (V,) per-region values for one gradient axis
    vertex_labels : (32768,) parcel label per vertex
                    (1-based; 0 = medial wall)
    n_vertices : int, default 32768 (fsLR-32k)

    Returns
    -------
    vertex_data : (32768,) vertex-wise map
    """
    vertex_data = np.zeros(n_vertices)
    valid = vertex_labels > 0
    vertex_data[valid] = parcel_values[vertex_labels[valid] - 1]
    return vertex_data


def vertexwise_correlation(map_a, map_b, mask=None):
    """Vertex-wise Pearson correlation between two surface maps."""
    if mask is not None:
        map_a, map_b = map_a[mask], map_b[mask]
    return float(pearsonr(map_a, map_b)[0])


def cross_similarity(grad_a, grad_b, labels_a, labels_b,
                     d_axes=None, n_vertices=32768):
    """
    CAS / CDS: per-axis vertex-wise Pearson correlation between two
    sets of gradients projected onto the fsLR-32k surface.

    Parameters
    ----------
    grad_a, grad_b : (V, d) region-level gradients
        (e.g., Glasser 360 vs Schaefer 400 for CAS, or HCP vs CHCP
         for CDS). Both must already be co-registered to a common
         template (e.g., via Procrustes alignment).
    labels_a, labels_b : parcel-to-vertex label mappings for each atlas.

    Returns
    -------
    sims : (d,) ndarray of per-axis surface correlations
    """
    d = grad_a.shape[1] if d_axes is None else d_axes
    sims = []
    for axis in range(d):
        va = parcels_to_surface(grad_a[:, axis], labels_a, n_vertices)
        vb = parcels_to_surface(grad_b[:, axis], labels_b, n_vertices)
        sims.append(vertexwise_correlation(va, vb))
    return np.array(sims)


# =============================================================
# 4. Example workflow
# =============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    V, N, d = 360, 30, 5

    # ---- Simulated gradients and Yeo-7 network assignment ----
    G = rng.normal(size=(V, d))
    net_names = ["VIS", "SOM", "DAN", "VAN", "LIM", "FPN", "DMN"]
    assignment = rng.choice(net_names, size=V)
    networks = {n: np.where(assignment == n)[0] for n in net_names}

    print("Step 1. Gradient spatial metrics (Eq. 9-13)")
    summary = summarize_axis_metrics(G, networks, summary_networks=net_names)
    for name, vals in summary.items():
        print(f"  {name:30s} per-axis: {np.round(vals, 4)}")

    print("\nStep 2. Test-retest reliability ICC(2,1)")
    s1 = [G + rng.normal(0, 0.05, G.shape) for _ in range(N)]
    s2 = [G + rng.normal(0, 0.05, G.shape) for _ in range(N)]
    iccs = test_retest_reliability(s1, s2)
    print(f"  Per-axis ICC: {np.round(iccs, 3)}")

    print("\nStep 3. CAS / CDS example (simulated labels)")
    labels = rng.integers(1, V + 1, size=32768)
    G2 = G + rng.normal(0, 0.05, G.shape)
    sims = cross_similarity(G, G2, labels, labels)
    print(f"  Cross-similarity per axis: {np.round(sims, 4)}")
