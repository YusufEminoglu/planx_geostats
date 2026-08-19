# -*- coding: utf-8 -*-
"""Engines for the V1.0 advanced spatial statistics tools.

These four methods (Geary's C, Join Count Statistics, Global Lee's L, and
the Geodetector Q-statistic) all use permutation (Monte Carlo) inference
rather than closed-form asymptotic variance formulas. Cliff & Ord's exact
randomization variance for Geary's C and for Join Counts, and the
classical F-approximation for the Geodetector Q-statistic, all carry
assumptions (e.g. no within-stratum spatial autocorrelation for Q; a
symmetric weights matrix for the join-count variance) that are easy to get
subtly wrong and hard to verify from memory. Permutation inference sidesteps
that risk entirely - shuffle, recompute, compare to the empirical null - and
is standard, citable practice for exactly this reason (see each function's
docstring for the specific reference).

Local Geary's C and the Colocation Quotient (added for Group 03) follow the
same permutation-inference convention, reusing this plugin's existing
conditional-permutation pattern from Bivariate Local Moran's I (see
stats_engines.py::calculate_bivariate_local_moran) for the local statistic.
SKATER (also added for Group 03) has no significance test in the classical
sense - like the plugin's existing Multivariate Clustering (K-Means), it
is a partitioning method evaluated by within-cluster sum of squares, not a
hypothesis test - so it reports SSD diagnostics only, matching
calculate_kmeans()'s existing precedent.

Group 05's three additions (Lagrange Multiplier Diagnostics, Spatial
Durbin Model, Eigenvector Spatial Filtering) are different: LM
Diagnostics IS the formal, citable closed-form asymptotic test the rest
of this module has been deliberately avoiding elsewhere - permutation is
not a substitute here, since the whole point of the tool is to reproduce
the standard Anselin, Bera, Florax & Yoon (1996) test that the
literature, GeoDa, and PySAL's spreg all report. Given the higher
correctness stakes of a closed-form formula, calculate_lm_diagnostics()
is validated against a hard mathematical invariant (LM-lag + Robust
LM-error == LM-error + Robust LM-lag, both equal the joint SARMA
statistic) as well as directional synthetic DGP checks, not just
plausibility - see the module's test notes in the project handoff. SDM
and ESF are both estimated by direct linear algebra (2SLS-style /
eigen-decomposition), not permutation or iteration, matching this
plugin's existing OLS/GWR engines' style.
"""
from __future__ import annotations

import heapq

import numpy as np

from .stats_engines import calculate_bivariate_lee_l


def calculate_geary_c(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Calculates Geary's C spatial autocorrelation via permutation inference.

    Geary's C (Geary, 1954; Cliff & Ord, 1981) is the sum-of-squared-
    differences alternative to Moran's I:
        C = [(N-1) * sum_i sum_j w_ij (y_i - y_j)^2] / [2 * S0 * sum_i (y_i - ybar)^2]
    Unlike Moran's I, C < 1 indicates positive spatial autocorrelation
    (similar values cluster) and C > 1 indicates negative autocorrelation -
    the opposite direction convention.

    Returns:
        A dict with observed_c, expected_c (= 1.0 under CSR), permuted_mean,
        permuted_std, z_score, p_value.
    """
    n = len(y)
    if n <= 3:
        raise ValueError("Geary's C requires at least 4 observations.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    i_idx: list[int] = []
    j_idx: list[int] = []
    w_list: list[float] = []
    for i, fid in enumerate(id_order):
        for j_fid, w in zip(neighbors.get(fid, []), weights.get(fid, [])):
            if j_fid in id_to_idx:
                i_idx.append(i)
                j_idx.append(id_to_idx[j_fid])
                w_list.append(w)

    if not i_idx:
        raise ValueError("No spatial neighbors found. Cannot compute Geary's C.")

    i_arr = np.array(i_idx)
    j_arr = np.array(j_idx)
    w_arr = np.array(w_list)
    s0 = float(np.sum(w_arr))

    def _geary_c(values: np.ndarray) -> float:
        y_mean = np.mean(values)
        denom = np.sum((values - y_mean) ** 2)
        if denom == 0 or s0 == 0:
            return 1.0
        numerator = np.sum(w_arr * (values[i_arr] - values[j_arr]) ** 2)
        return float(((n - 1) * numerator) / (2.0 * s0 * denom))

    observed_c = _geary_c(y)

    rng = np.random.default_rng(seed)
    perm_c = np.empty(permutations)
    for p in range(permutations):
        perm_c[p] = _geary_c(rng.permutation(y))

    perm_mean = float(np.mean(perm_c))
    perm_std = float(np.std(perm_c))
    z_score = (observed_c - perm_mean) / perm_std if perm_std > 0 else 0.0
    extreme = int(np.sum(np.abs(perm_c - perm_mean) >= abs(observed_c - perm_mean)))
    p_value = (extreme + 1) / (permutations + 1)

    return {
        "observed_c": observed_c,
        "expected_c": 1.0,
        "permuted_mean": perm_mean,
        "permuted_std": perm_std,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "permuted_values": perm_c.tolist(),
    }


def calculate_join_counts(
    x_binary: np.ndarray,
    neighbors: dict[int, list[int]],
    id_order: list[int],
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Calculates BB/WW/BW join counts for a binary field via permutation inference.

    Join Count Statistics (Cliff & Ord, 1973, 1981) test whether a binary
    (0/1) categorical field is spatially clustered by counting how many
    neighbor pairs ("joins") share the same category (BB, WW) versus differ
    (BW), against the count expected under random spatial arrangement of
    the same category sizes. Uses unweighted (binary) adjacency from
    `neighbors` - the traditional Join Count convention - not
    row-standardized weights, and each undirected pair is counted once
    even if the underlying weight builder is not perfectly symmetric
    (e.g. KNN).

    Returns:
        A dict with total_joins, n1, n0, and a bb/ww/bw sub-dict each
        containing observed, permuted_mean, permuted_std, z_score, p_value.
    """
    x_binary = np.asarray(x_binary, dtype=float)
    n = len(x_binary)
    if n <= 3:
        raise ValueError("Join Count Statistics require at least 4 observations.")
    unique_vals = set(np.unique(x_binary).tolist())
    if not unique_vals.issubset({0.0, 1.0}):
        raise ValueError("Join Count Statistics require a binary (0/1) field.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    pair_set: set[tuple[int, int]] = set()
    for i, fid in enumerate(id_order):
        for j_fid in neighbors.get(fid, []):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                pair_set.add((min(i, j), max(i, j)))

    if not pair_set:
        raise ValueError("No spatial neighbors found. Cannot compute Join Count Statistics.")

    i_arr = np.array([pair[0] for pair in pair_set])
    j_arr = np.array([pair[1] for pair in pair_set])
    total_joins = len(pair_set)

    def _counts(values: np.ndarray) -> tuple[int, int, int]:
        xi = values[i_arr]
        xj = values[j_arr]
        bb = int(np.sum((xi == 1) & (xj == 1)))
        ww = int(np.sum((xi == 0) & (xj == 0)))
        bw = total_joins - bb - ww
        return bb, ww, bw

    observed_bb, observed_ww, observed_bw = _counts(x_binary)

    rng = np.random.default_rng(seed)
    perm_bb = np.empty(permutations)
    perm_ww = np.empty(permutations)
    perm_bw = np.empty(permutations)
    for p in range(permutations):
        bb, ww, bw = _counts(rng.permutation(x_binary))
        perm_bb[p] = bb
        perm_ww[p] = ww
        perm_bw[p] = bw

    def _summary(observed: int, perm_arr: np.ndarray) -> dict:
        mean = float(np.mean(perm_arr))
        std = float(np.std(perm_arr))
        z = (observed - mean) / std if std > 0 else 0.0
        extreme = int(np.sum(np.abs(perm_arr - mean) >= abs(observed - mean)))
        p_val = (extreme + 1) / (permutations + 1)
        return {
            "observed": int(observed),
            "permuted_mean": mean,
            "permuted_std": std,
            "z_score": float(z),
            "p_value": float(p_val),
        }

    return {
        "total_joins": total_joins,
        "n1": int(np.sum(x_binary == 1)),
        "n0": int(np.sum(x_binary == 0)),
        "bb": _summary(observed_bb, perm_bb),
        "ww": _summary(observed_ww, perm_ww),
        "bw": _summary(observed_bw, perm_bw),
    }


def calculate_global_lee_l(
    x_values: np.ndarray,
    y_values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Calculates the global Lee's L bivariate spatial association statistic.

    Lee (2001) defines L(x,y) as a spatial smoothing of Pearson's r; under
    row-standardized weights (this plugin's convention) it reduces to the
    mean of the local l_i values already computed by
    calculate_bivariate_lee_l() in stats_engines.py - the same local-to-
    global decomposition relationship Local Moran's I has to Global
    Moran's I. Uses permutation inference (shuffle Y, holding the spatial
    structure and X fixed) rather than Lee's own asymptotic variance,
    which assumes normality of both fields.

    Returns:
        A dict with observed_l, permuted_mean, permuted_std, z_score, p_value.
    """
    n = len(x_values)
    if n <= 3:
        raise ValueError("Global Lee's L requires at least 4 observations.")

    def _global_l(x_arr: np.ndarray, y_arr: np.ndarray) -> float:
        local_l, _lag, _classes = calculate_bivariate_lee_l(x_arr, y_arr, neighbors, weights, id_order)
        return float(np.mean(local_l))

    observed_l = _global_l(x_values, y_values)

    rng = np.random.default_rng(seed)
    perm_l = np.empty(permutations)
    for p in range(permutations):
        perm_l[p] = _global_l(x_values, rng.permutation(y_values))

    perm_mean = float(np.mean(perm_l))
    perm_std = float(np.std(perm_l))
    z_score = (observed_l - perm_mean) / perm_std if perm_std > 0 else 0.0
    extreme = int(np.sum(np.abs(perm_l - perm_mean) >= abs(observed_l - perm_mean)))
    p_value = (extreme + 1) / (permutations + 1)

    return {
        "observed_l": observed_l,
        "permuted_mean": perm_mean,
        "permuted_std": perm_std,
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


def bin_into_quantiles(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assigns each value to one of n_bins quantile-based strata (0-indexed).

    Used to turn a continuous field into a categorical stratification for
    calculate_geodetector_q(), since a ready-made low-cardinality category
    field is the exception rather than the rule in real planning data.
    """
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        raise ValueError("Field has no variation; cannot bin into quantiles.")
    return np.digitize(values, edges[1:-1], right=True)


def calculate_geodetector_q(
    y: np.ndarray,
    strata: np.ndarray,
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Calculates the Geodetector Q-statistic for spatial stratified heterogeneity.

        q = 1 - [sum_h (N_h * var_h)] / [N * var_total]

    (Wang et al., 2010, IJGIS; Wang, Zhang & Fu, 2016, Ecological
    Indicators - the canonical Q-statistic paper). Tests whether a
    categorical stratification explains variance in a continuous field - a
    different paradigm from weights-based spatial autocorrelation; no
    spatial weights matrix is used at all. Uses permutation inference
    (shuffle strata labels, holding category sizes fixed) rather than the
    classical F-approximation, which assumes no within-stratum spatial
    autocorrelation - an assumption that rarely holds for planning data
    and that the F-test is known to be sensitive to.

    Returns:
        A dict with q_statistic, n_strata, permuted_mean, permuted_std,
        z_score, p_value (one-tailed: high q is the interesting direction),
        and per-stratum n/mean/std detail.
    """
    n = len(y)
    if n <= 3:
        raise ValueError("Geodetector Q-statistic requires at least 4 observations.")

    unique_strata = np.unique(strata)
    if len(unique_strata) < 2:
        raise ValueError("Geodetector Q-statistic requires at least 2 strata.")

    total_var = float(np.var(y))

    def _q(values: np.ndarray, labels: np.ndarray) -> float:
        if total_var == 0:
            return 0.0
        within_sum = 0.0
        for stratum in unique_strata:
            mask = labels == stratum
            n_h = int(np.sum(mask))
            if n_h == 0:
                continue
            within_sum += n_h * np.var(values[mask])
        return float(1.0 - (within_sum / (n * total_var)))

    observed_q = _q(y, strata)

    rng = np.random.default_rng(seed)
    perm_q = np.empty(permutations)
    for p in range(permutations):
        perm_q[p] = _q(y, rng.permutation(strata))

    perm_mean = float(np.mean(perm_q))
    perm_std = float(np.std(perm_q))
    z_score = (observed_q - perm_mean) / perm_std if perm_std > 0 else 0.0
    extreme = int(np.sum(perm_q >= observed_q))
    p_value = (extreme + 1) / (permutations + 1)

    per_stratum = []
    for stratum in unique_strata:
        mask = strata == stratum
        per_stratum.append({
            "stratum": int(stratum),
            "n": int(np.sum(mask)),
            "mean": float(np.mean(y[mask])),
            "std": float(np.std(y[mask])),
        })

    return {
        "q_statistic": observed_q,
        "n_strata": int(len(unique_strata)),
        "permuted_mean": perm_mean,
        "permuted_std": perm_std,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "per_stratum": per_stratum,
    }


class _LCG:
    """Lightweight deterministic PRNG for tight per-feature permutation loops.

    Matches the codebase's existing precedent in
    stats_engines.py::calculate_bivariate_local_moran - a per-feature x
    per-permutation nested loop calls this many more times than
    np.random.default_rng can service without per-call overhead dominating
    runtime.
    """

    __slots__ = ("_s",)
    _A = 6364136223846793005
    _C = 1442695040888963407
    _M = (1 << 64) - 1

    def __init__(self, seed):
        self._s = (int(seed) * 2 + 0x9E3779B97F4A7C15) & self._M

    def uniform(self, lo, hi):
        self._s = (self._A * self._s + self._C) & self._M
        frac = (self._s >> 11) / float(1 << 53)
        return lo + (hi - lo) * frac

    def shuffle(self, arr):
        for i in range(len(arr) - 1, 0, -1):
            j = int(self.uniform(0, i + 1))
            arr[i], arr[j] = arr[j], arr[i]


def calculate_local_geary_c(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    permutations: int = 999,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Calculates Anselin's (2019) Local Geary's C via conditional permutation.

        c_i = sum_j w_ij (y_i - y_j)^2

    A small c_i (relative to the permutation null) indicates the feature is
    similar to its neighbors (locally clustered); a large c_i indicates
    dissimilarity (a local outlier) - the local complement to the global
    Geary's C implemented above, following the same reversed-direction
    convention. Classification into HH/LL/HL/LH quadrants uses the sign of
    the feature's own deviation from the mean and its spatial lag's
    deviation, exactly mirroring calculate_local_moran()'s classification
    rule (so the two tools' output vocabularies are directly comparable).
    Significance is assessed by conditional permutation: y_i is held fixed
    and its neighbor set is resampled from the remaining N-1 observations,
    matching calculate_bivariate_local_moran()'s existing convention
    (Anselin, 1995, p.97, on conditional vs. total randomization for local
    statistics).

    Returns:
        A tuple of:
          - local_c: NumPy array of local Geary's C values
          - z_scores: NumPy array of permutation-based z-scores
          - p_values: NumPy array of two-tailed pseudo p-values
          - quadrants: List of strings ('HH', 'LL', 'HL', 'LH', 'Not Significant')
    """
    n = len(y)
    local_c = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        return local_c, z_scores, p_values, quadrants

    y_mean = float(np.mean(y))
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    spatial_lags = np.zeros(n)
    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])
                valid_w.append(f_weights[j])

        w_sum = sum(valid_w)
        if w_sum == 0:
            continue

        w_arr = np.array(valid_w) / w_sum
        neighbor_vals = y[valid_neigh_indices]
        spatial_lags[idx] = np.sum(w_arr * neighbor_vals)
        local_c[idx] = np.sum(w_arr * (y[idx] - neighbor_vals) ** 2)

    rng = _LCG(seed)

    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])
                valid_w.append(f_weights[j])

        w_sum = sum(valid_w)
        if w_sum == 0:
            continue

        w_arr = np.array(valid_w) / w_sum
        observed_c = local_c[idx]
        num_neighbors = len(valid_neigh_indices)
        other_indices = [i for i in range(n) if i != idx]

        perm_c_vals = np.zeros(permutations)
        for p in range(permutations):
            shuffled_others = list(other_indices)
            rng.shuffle(shuffled_others)
            perm_neighbor_indices = shuffled_others[:num_neighbors]
            perm_neighbor_vals = y[perm_neighbor_indices]
            perm_c_vals[p] = np.sum(w_arr * (y[idx] - perm_neighbor_vals) ** 2)

        mean_perm = float(np.mean(perm_c_vals))
        std_perm = float(np.std(perm_c_vals))
        z_scores[idx] = (observed_c - mean_perm) / std_perm if std_perm > 0 else 0.0

        extreme_count = np.sum(np.abs(perm_c_vals - mean_perm) >= abs(observed_c - mean_perm))
        p_val = (extreme_count + 1) / (permutations + 1)
        p_values[idx] = p_val

        if p_val < 0.05:
            high_val = y[idx] > y_mean
            high_lag = spatial_lags[idx] > y_mean
            if high_val and high_lag:
                quadrants[idx] = "HH"
            elif not high_val and not high_lag:
                quadrants[idx] = "LL"
            elif high_val and not high_lag:
                quadrants[idx] = "HL"
            elif not high_val and high_lag:
                quadrants[idx] = "LH"

    return local_c, z_scores, p_values, quadrants


def calculate_colocation_quotient(
    categories: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    category_a,
    category_b,
    k_neighbors: int = 5,
    permutations: int = 999,
    seed: int = 42,
) -> dict:
    """Calculates the K-nearest-neighbor Colocation Quotient CLQ(A->B).

    CLQ(A->B) measures whether category B tends to be spatially co-located
    with category A more (CLQ > 1), less (CLQ < 1), or exactly as expected
    (CLQ = 1) given B's overall prevalence in the dataset (Leslie &
    Kronenfeld, 2011), extended from the original nearest-neighbor
    formulation to K nearest neighbors following the local-CLQ convention
    of Cromley, Hanink & Bentley (2014):

        local_CLQ_i(A->B) = (NN_B(i) / K) / (N_B / (N-1)),  for each A-point i
        CLQ(A->B) = mean_i( local_CLQ_i(A->B) )

    where NN_B(i) counts how many of point i's K nearest neighbors are
    labeled B. CLQ is asymmetric: CLQ(A->B) generally differs from
    CLQ(B->A) - this is the statistic's key advantage over symmetric
    measures like Ripley's cross-K. Self-colocation (category_a ==
    category_b) is a valid, meaningful special case. Significance is
    assessed by permutation: category labels are randomly reshuffled
    across the fixed point locations 999 times by default, and the
    observed CLQ is compared to this empirical null.

    Returns:
        A dict with clq, n_a, n_b, n_total, k_neighbors, permuted_mean,
        permuted_std, z_score, p_value, and local_clq (per-A-point array,
        in the same order as the category-A subset of the input arrays).
    """
    n = len(categories)
    if n <= k_neighbors:
        raise ValueError("Colocation Quotient requires more observations than K.")

    coords = np.column_stack((x_coords, y_coords))
    cats = np.asarray(categories)

    n_a = int(np.sum(cats == category_a))
    n_b = int(np.sum(cats == category_b))
    if n_a == 0:
        raise ValueError(f"No features found in category A ('{category_a}').")
    if n_b == 0:
        raise ValueError(f"No features found in category B ('{category_b}').")

    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="auto").fit(coords)
        _, neighbor_idx = nbrs.kneighbors(coords)
        neighbor_idx = neighbor_idx[:, 1:]
    except ImportError:
        neighbor_idx = np.zeros((n, k_neighbors), dtype=int)
        chunk_size = 1000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_coords = coords[start:end]
            d = np.sqrt(((chunk_coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
            for i in range(start, end):
                d[i - start, i] = np.inf
            neighbor_idx[start:end] = np.argsort(d, axis=1)[:, :k_neighbors]

    def _clq(cat_array: np.ndarray) -> tuple[float, np.ndarray]:
        a_indices = np.where(cat_array == category_a)[0]
        if len(a_indices) == 0:
            return 0.0, np.zeros(0)
        b_prevalence = np.sum(cat_array == category_b) / (n - 1)
        if b_prevalence == 0:
            return 0.0, np.zeros(len(a_indices))
        local_clq = np.array([
            (np.sum(cat_array[neighbor_idx[i]] == category_b) / k_neighbors) / b_prevalence
            for i in a_indices
        ])
        return float(np.mean(local_clq)), local_clq

    observed_clq, local_clq = _clq(cats)

    rng = np.random.default_rng(seed)
    perm_clq = np.empty(permutations)
    for p in range(permutations):
        shuffled = rng.permutation(cats)
        perm_clq[p], _ = _clq(shuffled)

    perm_mean = float(np.mean(perm_clq))
    perm_std = float(np.std(perm_clq))
    z_score = (observed_clq - perm_mean) / perm_std if perm_std > 0 else 0.0
    extreme = int(np.sum(np.abs(perm_clq - perm_mean) >= abs(observed_clq - perm_mean)))
    p_value = (extreme + 1) / (permutations + 1)

    return {
        "clq": observed_clq,
        "n_a": n_a,
        "n_b": n_b,
        "n_total": n,
        "k_neighbors": k_neighbors,
        "permuted_mean": perm_mean,
        "permuted_std": perm_std,
        "z_score": float(z_score),
        "p_value": float(p_value),
        "local_clq": local_clq,
    }


def calculate_skater(
    data: np.ndarray,
    neighbors: dict[int, list[int]],
    id_order: list[int],
    k_clusters: int,
) -> tuple[np.ndarray, float]:
    """Performs SKATER spatially constrained clustering (Assunção et al., 2006).

    Builds a minimum spanning tree (Prim's algorithm) over the spatial
    contiguity graph, with edge weights equal to squared Euclidean distance
    between Z-score-standardized attribute vectors, then greedily removes
    the K-1 tree edges whose removal most reduces total within-cluster sum
    of squares (SSD) - producing K spatially CONTIGUOUS clusters, unlike
    the plugin's existing Multivariate Clustering (plain K-Means), which
    has no spatial-contiguity constraint at all. Ships as a native NumPy +
    stdlib heapq implementation (no optional spopt/libpysal dependency) so
    the tool works out of the box with zero extra installs, matching the
    reliability bar set by the plugin's other native engines.

    Returns:
        A tuple of (labels, total_ssd) where labels is a 0-indexed NumPy
        array of cluster assignments (aligned to id_order) and total_ssd is
        the sum of within-cluster sums of squares across all K clusters
        (the SKATER analogue of K-Means' WCSS).
    """
    n, p = data.shape
    if n < k_clusters:
        raise ValueError("Number of features must be >= k_clusters.")

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0.0] = 1.0
    z = (data - means) / stds

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    edge_set: set[tuple[int, int]] = set()
    for fid in id_order:
        i = id_to_idx[fid]
        for nid in neighbors.get(fid, []):
            j = id_to_idx.get(nid)
            if j is not None and i != j:
                edge_set.add((min(i, j), max(i, j)))

    if not edge_set:
        raise ValueError(
            "No spatial contiguity edges found. SKATER requires a connected "
            "neighborhood graph; try Queen contiguity or a larger K/distance band."
        )

    full_adj: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n)}
    for i, j in edge_set:
        dist = float(np.sum((z[i] - z[j]) ** 2))
        full_adj[i].append((dist, j))
        full_adj[j].append((dist, i))

    reached = [False] * n
    stack = [0]
    reached[0] = True
    reached_count = 1
    while stack:
        node = stack.pop()
        for _, nb in full_adj[node]:
            if not reached[nb]:
                reached[nb] = True
                reached_count += 1
                stack.append(nb)
    if reached_count < n:
        raise ValueError(
            "The spatial contiguity graph is disconnected "
            f"({reached_count} of {n} features reachable from a single connected "
            "component). SKATER requires full connectivity; widen the "
            "neighborhood definition (a larger K or distance band) so every "
            "feature is reachable."
        )

    visited = [False] * n
    heap: list[tuple[float, int, int]] = [(0.0, 0, -1)]
    mst_edges: list[tuple[float, int, int]] = []
    while heap and len(mst_edges) < n - 1:
        dist, node, parent = heapq.heappop(heap)
        if visited[node]:
            continue
        visited[node] = True
        if parent != -1:
            mst_edges.append((dist, parent, node))
        for d2, nb in full_adj[node]:
            if not visited[nb]:
                heapq.heappush(heap, (d2, nb, node))

    def _ssd(node_indices) -> float:
        pts = z[list(node_indices)]
        centroid = np.mean(pts, axis=0)
        return float(np.sum((pts - centroid) ** 2))

    def _split(nodes: set, edges: list, remove_idx: int) -> tuple[set, set]:
        local_adj: dict[int, list[int]] = {node: [] for node in nodes}
        for k, (_, a, b) in enumerate(edges):
            if k == remove_idx:
                continue
            local_adj[a].append(b)
            local_adj[b].append(a)
        start = edges[remove_idx][1]
        comp_a = {start}
        stack2 = [start]
        while stack2:
            node = stack2.pop()
            for nb in local_adj[node]:
                if nb not in comp_a:
                    comp_a.add(nb)
                    stack2.append(nb)
        comp_b = nodes - comp_a
        return comp_a, comp_b

    clusters: list[set] = [set(range(n))]
    cluster_edges: list[list[tuple[float, int, int]]] = [mst_edges]
    cluster_ssd = [_ssd(clusters[0])]

    for _ in range(k_clusters - 1):
        best = None
        for c_idx in range(len(clusters)):
            nodes = clusters[c_idx]
            edges = cluster_edges[c_idx]
            if len(nodes) < 2 or not edges:
                continue
            current_ssd = cluster_ssd[c_idx]
            for e_idx in range(len(edges)):
                comp_a, comp_b = _split(nodes, edges, e_idx)
                ssd_a = _ssd(comp_a)
                ssd_b = _ssd(comp_b)
                reduction = current_ssd - (ssd_a + ssd_b)
                if best is None or reduction > best[0]:
                    best = (reduction, c_idx, e_idx, comp_a, comp_b, ssd_a, ssd_b)
        if best is None:
            break
        _, c_idx, e_idx, comp_a, comp_b, ssd_a, ssd_b = best
        old_edges = cluster_edges[c_idx]
        remaining_edges = [e for k, e in enumerate(old_edges) if k != e_idx]
        edges_a = [e for e in remaining_edges if e[1] in comp_a and e[2] in comp_a]
        edges_b = [e for e in remaining_edges if e[1] in comp_b and e[2] in comp_b]
        clusters[c_idx] = comp_a
        cluster_edges[c_idx] = edges_a
        cluster_ssd[c_idx] = ssd_a
        clusters.append(comp_b)
        cluster_edges.append(edges_b)
        cluster_ssd.append(ssd_b)

    labels = np.zeros(n, dtype=int)
    for c_idx, nodes in enumerate(clusters):
        for node in nodes:
            labels[node] = c_idx

    total_ssd = float(sum(cluster_ssd))
    return labels, total_ssd


def _dense_weights_matrix(neighbors: dict[int, list[int]], weights: dict[int, list[float]], id_order: list[int]) -> np.ndarray:
    """Builds a dense (row-standardized) N x N spatial weights matrix aligned to id_order."""
    n = len(id_order)
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    W = np.zeros((n, n))
    for i, fid in enumerate(id_order):
        for nid, w in zip(neighbors.get(fid, []), weights.get(fid, [])):
            j = id_to_idx.get(nid)
            if j is not None:
                W[i, j] = w
    return W


def _chi2_1df_sf(x: float) -> float:
    """Exact survival function for a chi-square(1) statistic: P(X > x) = 1 - erf(sqrt(x/2))."""
    if x <= 0:
        return 1.0
    import math
    return float(max(0.0, min(1.0, 1.0 - math.erf(math.sqrt(x / 2.0)))))


def _chi2_2df_sf(x: float) -> float:
    """Exact survival function for a chi-square(2) statistic: P(X > x) = exp(-x/2)."""
    if x <= 0:
        return 1.0
    import math
    return float(max(0.0, min(1.0, math.exp(-0.5 * x))))


def calculate_lm_diagnostics(
    y: np.ndarray,
    X_data: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> dict:
    """Calculates the Lagrange Multiplier diagnostics for spatial dependence
    (Anselin, Bera, Florax & Yoon, 1996) - the formal pre-test for choosing
    between Spatial Lag (SAR) and Spatial Error (SEM) specifications that
    this plugin's existing SAR/SEM tools already tell users to "consult"
    without providing.

    Fits OLS internally, then computes the classic LM-lag and LM-error
    tests plus their ROBUST counterparts (each robust to the presence of
    the other form of dependence), using the standard information-matrix
    decomposition:

        T = tr(W'W + W^2)                         (fixed, depends only on W)
        D = T + (WXb)' M (WXb) / sigma^2           (M = I - X(X'X)^-1 X')
        S_rho    = e'Wy / sigma^2
        S_lambda = e'We / sigma^2

        LM-lag         = S_rho^2 / D
        LM-error       = S_lambda^2 / T
        Robust LM-lag   = (S_rho - S_lambda)^2 / (D - T)
        Robust LM-error = (S_lambda - (T/D) S_rho)^2 / (T (1 - T/D))

    Each statistic is asymptotically chi-square(1) under its null. As a
    mathematical identity (used to validate this implementation),
    LM-lag + Robust-LM-error == LM-error + Robust-LM-lag, both equal the
    joint SARMA (chi-square(2)) statistic.

    Returns:
        A dict with lm_lag, lm_lag_p, lm_error, lm_error_p, rlm_lag,
        rlm_lag_p, rlm_error, rlm_error_p, sarma, sarma_p, and a
        `recommendation` string following Anselin's classification rule.
    """
    n = len(y)
    p = X_data.shape[1]
    X = np.column_stack((np.ones(n), X_data))

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    y_hat = X @ beta
    e = y - y_hat
    sigma2 = float(e @ e) / n
    if sigma2 <= 0:
        raise ValueError("Residual variance is zero; LM diagnostics require non-degenerate OLS residuals.")

    W = _dense_weights_matrix(neighbors, weights, id_order)
    We = W @ e
    Wy = W @ y
    WXb = W @ y_hat

    T = float(np.trace(W @ W) + np.trace(W.T @ W))
    if T <= 0:
        raise ValueError("Spatial weights trace term is zero; check the spatial weights configuration.")

    WXb_M_WXb = float(WXb @ WXb - (WXb @ X) @ xtx_inv @ (X.T @ WXb))
    D = T + WXb_M_WXb / sigma2
    if D <= T:
        D = T * 1.0001  # numerical guard: D must exceed T for the robust tests to be defined

    S_rho = float(e @ Wy) / sigma2
    S_lambda = float(e @ We) / sigma2

    lm_error = (S_lambda ** 2) / T
    lm_lag = (S_rho ** 2) / D
    rlm_lag = ((S_rho - S_lambda) ** 2) / (D - T)
    rlm_error = ((S_lambda - (T / D) * S_rho) ** 2) / (T * (1.0 - T / D))

    sarma = lm_lag + rlm_error  # == lm_error + rlm_lag, a validated identity

    lm_lag_p = _chi2_1df_sf(lm_lag)
    lm_error_p = _chi2_1df_sf(lm_error)
    rlm_lag_p = _chi2_1df_sf(rlm_lag)
    rlm_error_p = _chi2_1df_sf(rlm_error)
    sarma_p = _chi2_2df_sf(sarma)

    lag_sig = lm_lag_p < 0.05
    error_sig = lm_error_p < 0.05
    rlag_sig = rlm_lag_p < 0.05
    rerror_sig = rlm_error_p < 0.05

    if not lag_sig and not error_sig:
        recommendation = "Neither LM-lag nor LM-error is significant: OLS is adequate; no spatial specification is indicated by this diagnostic."
    elif lag_sig and not error_sig:
        recommendation = "LM-lag is significant and LM-error is not: a Spatial Lag (SAR) or Spatial Durbin (SDM) specification is indicated."
    elif error_sig and not lag_sig:
        recommendation = "LM-error is significant and LM-lag is not: a Spatial Error (SEM) specification is indicated."
    else:
        # both significant: consult the robust versions
        if rlag_sig and not rerror_sig:
            recommendation = "Both LM tests are significant, but only Robust LM-lag survives: a Spatial Lag (SAR) or Spatial Durbin (SDM) specification is indicated."
        elif rerror_sig and not rlag_sig:
            recommendation = "Both LM tests are significant, but only Robust LM-error survives: a Spatial Error (SEM) specification is indicated."
        elif rlag_sig and rerror_sig:
            recommendation = "Both LM tests AND both robust LM tests are significant: consider a model with both spatial lag and spatial error components (e.g. Spatial Durbin Model), since neither pure specification fully resolves the dependence."
        else:
            recommendation = "Both LM tests are significant but neither robust test is: this is an unusual, weak-signal pattern - inspect the model specification and residual diagnostics before choosing a spatial model."

    return {
        "lm_lag": float(lm_lag),
        "lm_lag_p": float(lm_lag_p),
        "lm_error": float(lm_error),
        "lm_error_p": float(lm_error_p),
        "rlm_lag": float(rlm_lag),
        "rlm_lag_p": float(rlm_lag_p),
        "rlm_error": float(rlm_error),
        "rlm_error_p": float(rlm_error_p),
        "sarma": float(sarma),
        "sarma_p": float(sarma_p),
        "recommendation": recommendation,
    }


def calculate_spatial_durbin(
    y: np.ndarray,
    X_data: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    x_names: list[str],
) -> dict:
    """Estimates the Spatial Durbin Model (SDM) by Spatial Two-Stage Least
    Squares (Kelejian & Prucha, 1998).

        y = rho * Wy + X*beta + WX*theta + epsilon

    SDM lags both the dependent variable AND the explanatory variables,
    nesting Spatial Lag (SAR, theta=0) and a restricted form of Spatial
    Error (SEM, when theta = -rho*beta) as special cases (LeSage & Pace,
    2009). Wy is endogenous (correlated with the error term through the
    simultaneous spatial feedback), so OLS is inconsistent; this engine
    instruments Wy with [WX, W^2X] - the standard, well-established
    instrument set for the spatial lag family (Kelejian & Prucha, 1998;
    Lee, 2003) - avoiding the eigenvalue/log-determinant machinery a full
    Maximum Likelihood estimator would require, in favor of closed-form
    linear algebra with a much smaller correctness surface.

    Returns:
        A dict with coefficient names/estimates/standard errors/t-stats/
        p-values (Intercept, X variables, then WX variables, then rho),
        r2 (using the original, non-projected fitted values), n, k, and
        residuals.
    """
    n = len(y)
    p = X_data.shape[1]
    X = np.column_stack((np.ones(n), X_data))

    W = _dense_weights_matrix(neighbors, weights, id_order)
    Wy = W @ y
    WX = W @ X_data
    W2X = W @ WX

    # Full regressor matrix: [Intercept, X, WX, Wy]
    Z = np.column_stack((X, WX, Wy))
    # Instrument matrix: [Intercept, X, WX, W^2X]
    H = np.column_stack((X, WX, W2X))

    HtH_inv = np.linalg.pinv(H.T @ H)
    P_H = H @ HtH_inv @ H.T
    Z_hat = P_H @ Z

    ZtZhat_inv = np.linalg.pinv(Z_hat.T @ Z)
    delta = ZtZhat_inv @ (Z_hat.T @ y)

    y_pred = Z @ delta
    residuals = y - y_pred
    k = Z.shape[1]
    df_err = n - k
    if df_err <= 0:
        raise ValueError(f"Sample size ({n}) must exceed the number of SDM parameters ({k}).")

    ss_res = float(residuals @ residuals)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    sigma2 = ss_res / df_err
    cov_delta = sigma2 * np.linalg.pinv(Z_hat.T @ Z_hat)
    se_delta = np.sqrt(np.maximum(0.0, np.diagonal(cov_delta)))

    import math
    t_stats = np.zeros(k)
    p_vals = np.ones(k)
    for j in range(k):
        if se_delta[j] > 0:
            t_stats[j] = delta[j] / se_delta[j]
            p_vals[j] = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stats[j]) / math.sqrt(2.0))))

    variable_names = ["Intercept"] + x_names + [f"W_{name}" for name in x_names] + ["rho (W_y)"]

    return {
        "coefficients": delta,
        "std_errors": se_delta,
        "t_statistics": t_stats,
        "p_values": p_vals,
        "variable_names": variable_names,
        "r2": float(r2),
        "n": n,
        "k": k,
        "df_err": df_err,
        "residuals": residuals,
        "rho": float(delta[-1]),
        "rho_p": float(p_vals[-1]),
    }


def select_esf_eigenvectors(
    y: np.ndarray,
    X_data: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    max_eigenvectors: int = 10,
    p_value_threshold: float = 0.1,
) -> dict:
    """Selects spatial-filter eigenvectors via Griffith's (2003) Eigenvector
    Spatial Filtering (ESF) forward stepwise procedure.

    Builds the symmetric, doubly-centered Moran eigenvector map (MEM)
    matrix MC = (I - 11'/n) W (I - 11'/n), symmetrized, and eigen-
    decomposes it - each eigenvector is a candidate spatial pattern, with
    eigenvalue proportional to the Moran's I achievable by that pattern
    under W (Tiefelsdorf & Griffith, 2007). Only eigenvectors with a
    positive eigenvalue (positive spatial autocorrelation candidates) are
    considered, per Griffith's standard practical restriction. Forward
    stepwise selection then adds, one at a time, whichever remaining
    candidate eigenvector is most correlated with the CURRENT residual of
    y ~ X + already-selected eigenvectors, stopping once the residual
    Global Moran's I is no longer significant (p >= p_value_threshold) or
    max_eigenvectors is reached - reusing this plugin's existing
    `residual_spatial_autocorrelation_summary` for the stopping check
    rather than a new formula.

    Returns:
        A dict with selected_indices (into the candidate eigenvector
        list), eigenvectors (N x k array of the selected eigenvectors, in
        selection order), eigenvalues (their corresponding eigenvalues),
        and n_candidates (how many positive-eigenvalue candidates existed).
    """
    from .analysis_diagnostics import residual_spatial_autocorrelation_summary

    n = len(y)
    W = _dense_weights_matrix(neighbors, weights, id_order)
    ones = np.ones((n, 1))
    centering = np.eye(n) - (ones @ ones.T) / n
    MC = centering @ W @ centering
    MC = (MC + MC.T) / 2.0

    eigvals, eigvecs = np.linalg.eigh(MC)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    positive_mask = eigvals > 1e-9
    candidate_vals = eigvals[positive_mask]
    candidate_vecs = eigvecs[:, positive_mask]
    n_candidates = int(np.sum(positive_mask))

    X = np.column_stack((np.ones(n), X_data))
    selected_cols: list[np.ndarray] = []
    selected_indices: list[int] = []
    selected_vals: list[float] = []
    remaining = list(range(n_candidates))

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta

    for _ in range(min(max_eigenvectors, n_candidates)):
        corr_summary = residual_spatial_autocorrelation_summary(residual, neighbors, weights, id_order)
        if corr_summary.get("available") and corr_summary.get("p_value") is not None and corr_summary["p_value"] >= p_value_threshold:
            break
        if not remaining:
            break

        corrs = [abs(float(np.corrcoef(candidate_vecs[:, idx], residual)[0, 1])) for idx in remaining]
        best_local = int(np.argmax(corrs))
        best_idx = remaining.pop(best_local)
        selected_indices.append(best_idx)
        selected_vals.append(float(candidate_vals[best_idx]))
        selected_cols.append(candidate_vecs[:, best_idx])

        E = np.column_stack(selected_cols)
        design = np.column_stack((X, E))
        design_inv = np.linalg.pinv(design.T @ design)
        coefs = design_inv @ design.T @ y
        residual = y - design @ coefs

    selected_matrix = np.column_stack(selected_cols) if selected_cols else np.zeros((n, 0))

    return {
        "selected_indices": selected_indices,
        "eigenvectors": selected_matrix,
        "eigenvalues": selected_vals,
        "n_candidates": n_candidates,
    }
