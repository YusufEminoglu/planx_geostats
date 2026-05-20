# -*- coding: utf-8 -*-
"""Spatial statistics engines for calculations."""
from __future__ import annotations

import logging
import math
import numpy as np

logger = logging.getLogger("PlanX-GeoStats")

# Try importing PySAL modules
HAS_PYQ = False
try:
    from esda.g import Gi_Local
    from esda.moran import Moran_Local
    import libpysal
    HAS_PYQ = True
except ImportError:
    pass


def calculate_getis_ord(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    star: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Getis-Ord Gi or Gi* statistics."""
    n = len(y)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    conf_bins = np.zeros(n, dtype=int)

    if n <= 1:
        return z_scores, p_values, conf_bins

    y_mean = np.mean(y)
    y_std = np.std(y)

    if y_std == 0:
        logger.warning("Standard deviation of the target field is zero. Gi* cannot be calculated.")
        return z_scores, p_values, conf_bins

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        valid_neigh_indices = [id_to_idx[nid] for nid in f_neighs if nid in id_to_idx]

        if star:
            if idx not in valid_neigh_indices:
                valid_neigh_indices.append(idx)

        num_neighbors = len(valid_neigh_indices)
        if num_neighbors == 0:
            continue

        w_row = np.ones(num_neighbors) / num_neighbors
        y_neigh = y[valid_neigh_indices]
        sum_w_x = np.sum(w_row * y_neigh)
        sum_w = np.sum(w_row)
        sum_w2 = np.sum(w_row ** 2)

        numerator = sum_w_x - y_mean * sum_w
        denom_term = (n * sum_w2 - (sum_w ** 2)) / (n - 1)
        if denom_term < 0:
            denom_term = 0.0
        denominator = y_std * math.sqrt(denom_term)

        if denominator > 0:
            z = numerator / denominator
            z_scores[idx] = z
            p = 1.0 - math.erf(abs(z) / math.sqrt(2.0))
            p_values[idx] = p

            if p < 0.01:
                conf_bins[idx] = 3 if z > 0 else -3
            elif p < 0.05:
                conf_bins[idx] = 2 if z > 0 else -2
            elif p < 0.10:
                conf_bins[idx] = 1 if z > 0 else -1
            else:
                conf_bins[idx] = 0

    return z_scores, p_values, conf_bins


def calculate_mean_center(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: np.ndarray | None = None
) -> tuple[float, float]:
    """Calculates the mean center of coordinate pairs."""
    if weights is None or len(weights) == 0:
        return float(np.mean(x_coords)), float(np.mean(y_coords))
    
    total_weight = np.sum(weights)
    if total_weight == 0:
        return float(np.mean(x_coords)), float(np.mean(y_coords))
        
    mean_x = np.sum(x_coords * weights) / total_weight
    mean_y = np.sum(y_coords * weights) / total_weight
    return float(mean_x), float(mean_y)


def calculate_central_feature(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: np.ndarray | None = None
) -> int:
    """Finds the index of the central feature based on minimum total distance."""
    n = len(x_coords)
    if n <= 1:
        return 0

    coords = np.column_stack((x_coords, y_coords))
    # Pairwise Euclidean distances
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    if weights is None or len(weights) == 0:
        dist_sums = dists.sum(axis=1)
    else:
        # Weighted distance sum
        dist_sums = (dists * weights[None, :]).sum(axis=1)

    return int(np.argmin(dist_sums))


def calculate_sde(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: np.ndarray | None = None,
    num_std: int = 1
) -> tuple[float, float, float, float, float]:
    """Calculates Standard Deviational Ellipse (SDE) parameters.

    Returns:
        A tuple of (mean_x, mean_y, rotation_angle_radians, semi_major_axis, semi_minor_axis)
    """
    n = len(x_coords)
    mean_x, mean_y = calculate_mean_center(x_coords, y_coords, weights)
    
    if n <= 2:
        return mean_x, mean_y, 0.0, 0.0, 0.0

    x_prime = x_coords - mean_x
    y_prime = y_coords - mean_y

    W = np.ones(n) if (weights is None or len(weights) == 0) else weights
    sum_w = np.sum(W)
    if sum_w == 0:
        W = np.ones(n)
        sum_w = n

    sum_x2 = np.sum(W * (x_prime ** 2))
    sum_y2 = np.sum(W * (y_prime ** 2))
    sum_xy = np.sum(W * x_prime * y_prime)

    # Calculate rotation angle theta
    # Using the standardPrincipal Orientation formula
    theta = 0.5 * np.arctan2(2 * sum_xy, sum_x2 - sum_y2)

    # Standard deviations along rotated axes
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    std_x = np.sqrt(np.sum(W * (x_prime * cos_t - y_prime * sin_t) ** 2) / sum_w)
    std_y = np.sqrt(np.sum(W * (x_prime * sin_t + y_prime * cos_t) ** 2) / sum_w)

    # Semi-major/minor axes scaling
    semi_x = num_std * std_x
    semi_y = num_std * std_y

    # Let semi_major be the larger one
    if semi_x >= semi_y:
        semi_major = semi_x
        semi_minor = semi_y
        angle = theta
    else:
        semi_major = semi_y
        semi_minor = semi_x
        angle = theta + np.pi / 2.0  # Align rotation to semi-major axis

    return mean_x, mean_y, float(angle), float(semi_major), float(semi_minor)


def calculate_local_moran(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Calculates Anselin Local Moran's I cluster and outlier diagnostics.

    Returns:
        A tuple of:
          - I_values: NumPy array of Moran's I indices (floats)
          - z_scores: NumPy array of z-scores (floats)
          - p_values: NumPy array of p-values (floats)
          - quadrants: List of strings ('HH', 'LL', 'HL', 'LH', 'Not Significant')
    """
    n = len(y)
    I_values = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        return I_values, z_scores, p_values, quadrants

    y_mean = np.mean(y)
    z = y - y_mean
    m2 = np.sum(z ** 2) / n

    if m2 == 0:
        return I_values, z_scores, p_values, quadrants

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    b2 = (n * np.sum(z ** 4)) / (np.sum(z ** 2) ** 2)  # Kurtosis

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
        w_sum2 = sum(w**2 for w in valid_w)

        if w_sum == 0:
            continue

        # Spatial lag
        spatial_lag = np.sum(np.array(valid_w) * z[valid_neigh_indices])
        I_i = (z[idx] / m2) * spatial_lag
        I_values[idx] = I_i

        # Expected value under randomization
        E_Ii = -w_sum / (n - 1)

        # Variance under randomization (Anselin 1995 formula)
        # Var(Ii) = w_i2 * (n - b2) / (n - 1) + (w_i^2 - w_i2) * (2b2 - n) / ((n - 1)(n - 2)) - E(Ii)^2
        var_term1 = (w_sum2 * (n - b2)) / (n - 1)
        
        if n > 2:
            var_term2 = ((w_sum**2 - w_sum2) * (2*b2 - n)) / ((n - 1) * (n - 2))
        else:
            var_term2 = 0.0
            
        var_Ii = var_term1 + var_term2 - (E_Ii ** 2)

        if var_Ii > 0:
            z_i = (I_i - E_Ii) / math.sqrt(var_Ii)
            z_scores[idx] = z_i
            p = 1.0 - math.erf(abs(z_i) / math.sqrt(2.0))
            p_values[idx] = p

            # Quadrant categorization (HH, LL, HL, LH)
            if p < 0.05:
                # Value relative to mean
                high_val = z[idx] > 0
                # Lag relative to mean
                high_lag = spatial_lag > 0

                if high_val and high_lag:
                    quadrants[idx] = "HH"
                elif not high_val and not high_lag:
                    quadrants[idx] = "LL"
                elif high_val and not high_lag:
                    quadrants[idx] = "HL"
                elif not high_val and high_lag:
                    quadrants[idx] = "LH"
            else:
                quadrants[idx] = "Not Significant"
        else:
            z_scores[idx] = 0.0
            p_values[idx] = 1.0
            quadrants[idx] = "Not Significant"

    return I_values, z_scores, p_values, quadrants


def _chi2_sf_approx(x: float, df: int) -> float:
    """Wilson-Hilferty transformation approximation for Chi-Square Survival Function (p-value)."""
    if x <= 0:
        return 1.0
    if df == 2:
        return float(math.exp(-0.5 * x))  # Exact for df=2
    
    # Wilson-Hilferty approximation: Chi2 to normal
    d = float(df)
    z = ((x / d) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * d))) / math.sqrt(2.0 / (9.0 * d))
    p_val = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, p_val)))


def calculate_ols(
    y: np.ndarray,
    X_data: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    x_names: list[str]
) -> dict:
    """Performs Ordinary Least Squares (OLS) regression and diagnostic tests.

    Args:
        y: 1D dependent variable array (n,)
        X_data: 2D independent variables array (n, p)
        neighbors: Weights neighbors dict
        weights: Weights values dict
        id_order: Feature IDs
        x_names: Names of independent variables

    Returns:
        A dictionary containing coefficient estimates, diagnostics, residuals, etc.
    """
    n = len(y)
    p = X_data.shape[1]
    
    # Add intercept column
    X = np.column_stack((np.ones(n), X_data))
    
    # Solve beta = (X.T * X)^-1 * X.T * Y
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ y
    except Exception as e:
        logger.error("Linear algebra inversion failed in OLS regression: %s", e)
        raise ValueError(f"Regression inversion failed: {e}")

    # Residuals
    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    # Variance of residuals
    df_err = n - p - 1
    if df_err <= 0:
        raise ValueError(f"Sample size ({n}) must be greater than number of variables ({p} + intercept).")
        
    s2 = ss_res / df_err
    std_residuals = residuals / math.sqrt(s2) if s2 > 0 else np.zeros(n)
    
    # Standard Errors of Coefficients
    cov_beta = s2 * xtx_inv
    se_beta = np.sqrt(np.maximum(0.0, np.diagonal(cov_beta)))
    
    # t-statistics and p-values
    t_stats = np.zeros(p + 1)
    p_vals = np.ones(p + 1)
    for j in range(p + 1):
        if se_beta[j] > 0:
            t_stats[j] = beta[j] / se_beta[j]
            # Normal approximation for t-dist (very accurate for large df)
            p_vals[j] = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stats[j]) / math.sqrt(2.0))))
        else:
            t_stats[j] = 0.0
            p_vals[j] = 1.0

    # Model R2 & Adj R2
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_err

    # --- DIAGNOSTIC 1: Jarque-Bera normality test ---
    s2_ml = ss_res / n
    if s2_ml > 0:
        skew = np.sum(residuals ** 3) / n / (s2_ml ** 1.5)
        kurt = np.sum(residuals ** 4) / n / (s2_ml ** 2)
        jb_stat = (n / 6.0) * (skew ** 2 + 0.25 * (kurt - 3.0) ** 2)
        jb_p = _chi2_sf_approx(jb_stat, df=2)
    else:
        jb_stat, jb_p = 0.0, 1.0

    # --- DIAGNOSTIC 2: Koenker's Breusch-Pagan heteroskedasticity test ---
    # Auxiliary regression: e^2 on X_data
    g = residuals ** 2
    g_mean = np.mean(g)
    g_tot = np.sum((g - g_mean) ** 2)
    
    bp_stat, bp_p = 0.0, 1.0
    if g_tot > 0:
        try:
            # Regress g on independent variables
            beta_aux = np.linalg.pinv(X.T @ X) @ X.T @ g
            g_pred = X @ beta_aux
            g_res = g - g_pred
            ss_aux_res = np.sum(g_res ** 2)
            r2_aux = 1.0 - (ss_aux_res / g_tot)
            bp_stat = n * r2_aux
            bp_p = _chi2_sf_approx(bp_stat, df=p)
        except Exception:
            pass

    # --- DIAGNOSTIC 3: Moran's I on Residuals ---
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    spatial_lag_e = np.zeros(n)
    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])
        lag_sum = 0.0
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                lag_sum += f_weights[j] * residuals[id_to_idx[nid]]
        spatial_lag_e[idx] = lag_sum

    if ss_res > 0:
        moran_i = np.sum(residuals * spatial_lag_e) / ss_res
    else:
        moran_i = 0.0

    # Return OLS results dictionary
    return {
        "coefficients": beta,
        "std_errors": se_beta,
        "t_statistics": t_stats,
        "p_values": p_vals,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "p": p,
        "df_err": df_err,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "jarque_bera": (jb_stat, jb_p),
        "breusch_pagan": (bp_stat, bp_p),
        "residuals_moran": moran_i,
        "variable_names": ["Intercept"] + x_names
    }


def calculate_global_moran(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int]
) -> tuple[float, float, float, float, float]:
    """Calculates Global Moran's I spatial autocorrelation.

    Returns:
        A tuple of (moran_i, expected_i, variance, z_score, p_value)
    """
    n = len(y)
    if n <= 3:
        raise ValueError("Global Moran's I requires at least 4 observations.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    y_mean = np.mean(y)
    z = y - y_mean
    sum_z2 = np.sum(z ** 2)
    sum_z4 = np.sum(z ** 4)

    if sum_z2 == 0:
        return 0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0

    S0 = 0.0
    w_row_sums = np.zeros(n)
    w_col_sums = np.zeros(n)

    # First pass: compute S0, row sums, and column sums
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                S0 += w
                w_row_sums[i] += w
                w_col_sums[j] += w

    if S0 == 0:
        return 0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0

    # Second pass: compute S1
    S1 = 0.0
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w_ij in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                w_ji = 0.0
                j_neighs = neighbors.get(j_fid, [])
                j_w_list = weights.get(j_fid, [])
                if fid in j_neighs:
                    w_ji = j_w_list[j_neighs.index(fid)]
                S1 += (w_ij + w_ji) ** 2
    S1 = 0.5 * S1

    # Compute S2
    S2 = np.sum((w_row_sums + w_col_sums) ** 2)

    # Calculate Moran's I
    numerator = 0.0
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                numerator += w * z[i] * z[j]

    moran_i = (n / S0) * (numerator / sum_z2)

    # Expected value
    expected_i = -1.0 / (n - 1)

    # Kurtosis term D
    D = (n * sum_z4) / (sum_z2 ** 2)

    # Variance under randomization
    num_var = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2) - D * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    den_var = (n - 1) * (n - 2) * (n - 3) * S0 ** 2
    
    variance = num_var / den_var - (expected_i ** 2) if den_var > 0 else 0.0

    return float(moran_i), float(expected_i), float(variance), float(z_score), float(p_value)


def calculate_average_nearest_neighbor(
    x: np.ndarray,
    y: np.ndarray,
    study_area: float | None = None
) -> tuple[float, float, float, float, float, float]:
    """Calculates Average Nearest Neighbor statistics.

    Returns:
        A tuple of (observed_mean, expected_mean, nn_ratio, z_score, p_value, study_area)
    """
    n = len(x)
    if n <= 1:
        raise ValueError("Average Nearest Neighbor requires at least 2 points.")

    coords = np.column_stack((x, y))

    # Try using scikit-learn KDTree for maximum performance, fallback to vectorized NumPy
    try:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        nn_dists = distances[:, 1]
    except ImportError:
        # Vectorized chunked NumPy distance finder to protect memory
        nn_dists = np.zeros(n)
        chunk_size = 1000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_coords = coords[start:end]
            d = np.sqrt(((chunk_coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
            for i in range(start, end):
                d[i - start, i] = np.inf
            nn_dists[start:end] = np.min(d, axis=1)

    observed_mean = float(np.mean(nn_dists))

    # Fallback to minimum bounding box area if study area is not provided
    if study_area is None or study_area <= 0:
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        w = max_x - min_x
        h = max_y - min_y
        study_area = float(w * h) if (w * h > 0) else 1.0

    density = n / study_area
    expected_mean = 0.5 / math.sqrt(density)

    # Standard error
    se = 0.26136 / math.sqrt(n * density)

    nn_ratio = observed_mean / expected_mean if expected_mean > 0 else 1.0
    z_score = (observed_mean - expected_mean) / se if se > 0 else 0.0
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    return observed_mean, expected_mean, nn_ratio, z_score, p_value, study_area


def calculate_standard_distance(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Calculates Standard Distance and mean center.

    Returns:
        A tuple of (mean_x, mean_y, standard_distance)
    """
    n = len(x_coords)
    if n == 0:
        return 0.0, 0.0, 0.0

    mean_x, mean_y = calculate_mean_center(x_coords, y_coords, weights)

    if weights is None or len(weights) == 0:
        var_x = np.sum((x_coords - mean_x) ** 2) / n
        var_y = np.sum((y_coords - mean_y) ** 2) / n
    else:
        sum_w = np.sum(weights)
        if sum_w == 0:
            var_x = np.sum((x_coords - mean_x) ** 2) / n
            var_y = np.sum((y_coords - mean_y) ** 2) / n
        else:
            var_x = np.sum(weights * (x_coords - mean_x) ** 2) / sum_w
            var_y = np.sum(weights * (y_coords - mean_y) ** 2) / sum_w

    std_distance = math.sqrt(var_x + var_y)
    return mean_x, mean_y, std_distance


def calculate_gwr(
    y: np.ndarray,
    X_data: np.ndarray,
    coords: np.ndarray,
    bandwidth: float,
    kernel_type: str = "adaptive_bisquare"
) -> dict:
    """Performs Geographically Weighted Regression (GWR) analysis.

    Args:
        y: Dependent variable (n,)
        X_data: Independent variables (n, p)
        coords: Centroid coordinates (n, 2)
        bandwidth: Kernel bandwidth (distance or neighbor count)
        kernel_type: fixed_gaussian, fixed_bisquare, or adaptive_bisquare

    Returns:
        A dictionary containing GWR coefficients, errors, local R2, and statistics.
    """
    n = len(y)
    p = X_data.shape[1]

    # Add intercept column
    X = np.column_stack((np.ones(n), X_data))

    # Initialize results
    local_beta = np.zeros((n, p + 1))
    local_se = np.zeros((n, p + 1))
    local_t = np.zeros((n, p + 1))
    y_pred = np.zeros(n)
    local_r2 = np.zeros(n)

    # Compute distance matrix
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    for i in range(n):
        dists = dists_matrix[i]

        # Calculate weights based on kernel type
        if kernel_type == "fixed_gaussian":
            w = np.exp(-0.5 * (dists / bandwidth) ** 2)
        elif kernel_type == "fixed_bisquare":
            w = np.zeros(n)
            mask = dists < bandwidth
            w[mask] = (1.0 - (dists[mask] / bandwidth) ** 2) ** 2
        elif kernel_type == "adaptive_bisquare":
            k = int(bandwidth)
            sorted_dists = np.sort(dists)
            d_k = sorted_dists[min(k - 1, n - 1)]
            w = np.zeros(n)
            if d_k > 0:
                mask = dists < d_k
                w[mask] = (1.0 - (dists[mask] / d_k) ** 2) ** 2
            else:
                w[dists == 0] = 1.0
        else:
            w = np.ones(n)

        sum_w = np.sum(w)
        if sum_w == 0:
            w = np.ones(n)
            sum_w = n

        # Solve local regression: beta_i = (X.T * W * X)^-1 * X.T * W * Y
        try:
            xtw = X.T * w
            xtwx = xtw @ X
            xtwx_inv = np.linalg.pinv(xtwx)
            beta_i = xtwx_inv @ xtw @ y
            local_beta[i] = beta_i
            y_pred[i] = X[i] @ beta_i

            # Standard errors and t-statistics
            res_i = y - (X @ beta_i)
            df_i = sum_w - p - 1
            if df_i > 0:
                s2_i = np.sum(w * (res_i ** 2)) / df_i
                cov_beta_i = s2_i * xtwx_inv
                se_beta_i = np.sqrt(np.maximum(0.0, np.diagonal(cov_beta_i)))
                local_se[i] = se_beta_i
                for j in range(p + 1):
                    if se_beta_i[j] > 0:
                        local_t[i, j] = beta_i[j] / se_beta_i[j]

            # Local R2
            y_w_mean = np.sum(w * y) / sum_w
            tss_i = np.sum(w * (y - y_w_mean) ** 2)
            rss_i = np.sum(w * (res_i ** 2))
            local_r2[i] = 1.0 - (rss_i / tss_i) if tss_i > 0 else 1.0
        except Exception:
            pass

    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    global_r2 = 1.0 - (rss / tss) if tss > 0 else 0.0

    # Effective degrees of freedom (Trace of Hat Matrix)
    tr_S = 0.0
    for i in range(n):
        try:
            if kernel_type == "fixed_gaussian":
                w = np.exp(-0.5 * (dists_matrix[i] / bandwidth) ** 2)
            elif kernel_type == "fixed_bisquare":
                w = np.zeros(n)
                mask = dists_matrix[i] < bandwidth
                w[mask] = (1.0 - (dists_matrix[i][mask] / bandwidth) ** 2) ** 2
            elif kernel_type == "adaptive_bisquare":
                k = int(bandwidth)
                d_k = np.sort(dists_matrix[i])[min(k - 1, n - 1)]
                w = np.zeros(n)
                if d_k > 0:
                    mask = dists_matrix[i] < d_k
                    w[mask] = (1.0 - (dists_matrix[i][mask] / d_k) ** 2) ** 2
                else:
                    w[dists_matrix[i] == 0] = 1.0
            else:
                w = np.ones(n)

            xtw = X.T * w
            xtwx_inv = np.linalg.pinv(xtw @ X)
            s_i = (X[i] @ xtwx_inv) @ xtw
            tr_S += s_i[i]
        except Exception:
            pass

    if tr_S <= 0:
        tr_S = float(p + 1)

    aicc = np.inf
    if n - tr_S - 2 > 0 and rss > 0:
        aicc = n * np.log(rss / n) + n * np.log(2 * np.pi) + n * (n + tr_S) / (n - 2 - tr_S)

    return {
        "local_beta": local_beta,
        "local_se": local_se,
        "local_t": local_t,
        "y_pred": y_pred,
        "residuals": residuals,
        "local_r2": local_r2,
        "rss": rss,
        "tss": tss,
        "r2": global_r2,
        "aicc": aicc,
        "effective_df": tr_S
    }


def calculate_median_center(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6
) -> tuple[float, float, float]:
    """Calculates Median Center using Weiszfeld's algorithm.

    Returns:
        A tuple of (median_x, median_y, total_distance)
    """
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(x[0]), float(y[0]), 0.0

    if weights is None or len(weights) == 0:
        weights = np.ones(n)

    # Initial guess: mean center
    cx, cy = calculate_mean_center(x, y, weights)

    for _ in range(max_iter):
        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dists = np.maximum(dists, 1e-12)

        inv_dists = weights / dists
        sum_inv = np.sum(inv_dists)

        if sum_inv == 0:
            break

        new_cx = np.sum(x * inv_dists) / sum_inv
        new_cy = np.sum(y * inv_dists) / sum_inv

        shift = math.sqrt((new_cx - cx) ** 2 + (new_cy - cy) ** 2)
        cx, cy = new_cx, new_cy

        if shift < tol:
            break

    total_dist = float(np.sum(weights * np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    return float(cx), float(cy), total_dist


def calculate_general_g(
    values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int]
) -> tuple[float, float, float, float, float]:
    """Calculates Getis-Ord General G statistics under randomization.

    Returns:
        A tuple of (observed_g, expected_g, variance, z_score, p_value)
    """
    n = len(id_order)
    if n < 4:
        raise ValueError("General G requires at least 4 features.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    x = np.array([float(values[i]) for i in range(n)])

    x_sum = np.sum(x)
    x_sum2 = np.sum(x ** 2)
    x_sum3 = np.sum(x ** 3)
    x_sum4 = np.sum(x ** 4)

    # Build dense weight matrix
    W = np.zeros((n, n))
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                W[i, j] = w

    np.fill_diagonal(W, 0.0)

    S0 = np.sum(W)
    A1 = np.sum(W ** 2)
    A2 = np.sum(W * W.T)

    row_sums = np.sum(W, axis=1)
    col_sums = np.sum(W, axis=0)

    A3 = np.sum(row_sums ** 2) + np.sum(col_sums ** 2)
    A4 = np.sum(row_sums * col_sums)
    A5 = S0 ** 2

    # Observed G
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += W[i, j] * x[i] * x[j]

    denominator = x_sum ** 2 - x_sum2
    if denominator == 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0

    observed_g = numerator / denominator
    expected_g = S0 / (n * (n - 1))

    # Permutations of x values
    S_22 = x_sum2 ** 2 - x_sum4
    S_211 = x_sum2 * (x_sum ** 2 - x_sum2) - 2.0 * x_sum * x_sum3 + 2.0 * x_sum4
    S_1111 = (x_sum ** 4) - 6.0 * (x_sum ** 2) * x_sum2 + 8.0 * x_sum * x_sum3 + 3.0 * (x_sum2 ** 2) - 6.0 * x_sum4

    term1 = (A1 + A2) * S_22 / (n * (n - 1))
    term2 = (2.0 * A3 + 4.0 * A4 - 4.0 * A1 - 4.0 * A2) * S_211 / (n * (n - 1) * (n - 2))
    term3 = (A5 - 2.0 * A3 - 4.0 * A4 + 3.0 * A1 + 3.0 * A2) * S_1111 / (n * (n - 1) * (n - 2) * (n - 3))

    E_A2 = term1 + term2 + term3
    E_G2 = E_A2 / (denominator ** 2)

    variance = E_G2 - (expected_g ** 2)
    if variance > 0:
        z_score = (observed_g - expected_g) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
    else:
        z_score = 0.0
        p_value = 1.0

    return float(observed_g), float(expected_g), float(variance), float(z_score), float(p_value)


def calculate_similarity_search(
    full_data: np.ndarray,
    target_indices: list[int],
    metric: str = "euclidean"
) -> np.ndarray:
    """Standardizes attributes and computes distance score from target feature profiles.

    Returns:
        An array of similarity scores for each feature in full_data.
    """
    n, p = full_data.shape
    if n == 0 or p == 0:
        return np.array([])

    # Z-score standardization
    means = np.mean(full_data, axis=0)
    stds = np.std(full_data, axis=0)
    stds[stds == 0.0] = 1.0  # avoid division by zero

    z_data = (full_data - means) / stds

    # Extract target profile (mean profile if multiple targets are selected)
    z_targets = z_data[target_indices]
    target_profile = np.mean(z_targets, axis=0)

    # Compute score based on selected distance metric
    if metric == "manhattan":
        scores = np.sum(np.abs(z_data - target_profile), axis=1)
    else:  # euclidean
        scores = np.sqrt(np.sum((z_data - target_profile) ** 2, axis=1))

    return scores


def calculate_distance_band_stats(
    x: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 1
) -> dict:
    """Calculates statistics for distance to the k-th nearest neighbor.

    Returns:
        A dictionary containing min, max, mean, median, p25, p75 values.
    """
    n = len(x)
    if n <= 1:
        raise ValueError("At least 2 points are required to compute distance bands.")

    coords = np.column_stack((x, y))

    # Compute distance matrix
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    k_dists = np.zeros(n)
    for i in range(n):
        sorted_d = np.sort(dists_matrix[i])
        k_idx = min(k_neighbors, n - 1)
        k_dists[i] = sorted_d[k_idx]

    return {
        "min": float(np.min(k_dists)),
        "max": float(np.max(k_dists)),
        "mean": float(np.mean(k_dists)),
        "median": float(np.median(k_dists)),
        "p25": float(np.percentile(k_dists, 25)),
        "p75": float(np.percentile(k_dists, 75))
    }


def calculate_kmeans(
    data: np.ndarray,
    k_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42
) -> tuple[np.ndarray, float]:
    """Performs K-Means clustering on feature attribute data.

    Returns:
        A tuple of (labels, wcss) where wcss is within-cluster sum of squares.
    """
    n, p = data.shape
    if n < k_clusters:
        raise ValueError("Number of data points must be greater than or equal to k_clusters.")

    # Z-score standardization
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0.0] = 1.0
    z_data = (data - means) / stds

    # K-Means++ initialization
    rng = np.random.default_rng(seed)
    centroids = np.zeros((k_clusters, p))

    # Pick first centroid
    idx = rng.choice(n)
    centroids[0] = z_data[idx]

    for c_idx in range(1, k_clusters):
        # Distance squared to closest centroid
        dists_sq = np.min([np.sum((z_data - centroids[c]) ** 2, axis=1) for c in range(c_idx)], axis=0)
        
        # Avoid division by zero if all points are at the centroids
        sum_dists = np.sum(dists_sq)
        if sum_dists == 0:
            probs = np.ones(n) / n
        else:
            probs = dists_sq / sum_dists
            
        idx = rng.choice(n, p=probs)
        centroids[c_idx] = z_data[idx]

    # Lloyd's algorithm iterations
    labels = np.zeros(n, dtype=int)
    prev_wcss = np.inf

    for _ in range(max_iter):
        # Assign labels
        dists = np.array([np.sum((z_data - centroids[c]) ** 2, axis=1) for c in range(k_clusters)])
        labels = np.argmin(dists, axis=0)

        # Calculate WCSS
        wcss = 0.0
        for c in range(k_clusters):
            mask = (labels == c)
            if np.any(mask):
                wcss += np.sum((z_data[mask] - centroids[c]) ** 2)

        if abs(prev_wcss - wcss) < tol:
            break
        prev_wcss = wcss

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(k_clusters):
            mask = (labels == c)
            if np.any(mask):
                new_centroids[c] = np.mean(z_data[mask], axis=0)
            else:
                # Re-initialize empty cluster with a random point
                new_centroids[c] = z_data[rng.choice(n)]
        centroids = new_centroids

    return labels, float(wcss)


def calculate_linear_directional_mean(
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray
) -> tuple[float, float, float, float]:
    """Calculates Linear Directional Mean for line features.

    Returns:
        A tuple of (center_x, center_y, mean_angle_degrees, mean_length)
    """
    n = len(start_x)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Line midpoints
    mid_x = (start_x + end_x) / 2.0
    mid_y = (start_y + end_y) / 2.0

    center_x = float(np.mean(mid_x))
    center_y = float(np.mean(mid_y))

    # Line lengths
    dx = end_x - start_x
    dy = end_y - start_y
    lengths = np.sqrt(dx ** 2 + dy ** 2)
    lengths_safe = np.maximum(lengths, 1e-12)

    mean_length = float(np.mean(lengths))

    # Orientation angles (radians, compass-style: 0=North, clockwise)
    # atan2(dx, dy) gives compass bearing
    angles = np.arctan2(dx, dy)

    # Circular mean weighted by length
    sin_sum = np.sum(lengths_safe * np.sin(angles))
    cos_sum = np.sum(lengths_safe * np.cos(angles))

    mean_angle_rad = math.atan2(sin_sum, cos_sum)
    mean_angle_deg = math.degrees(mean_angle_rad) % 360.0

    return center_x, center_y, mean_angle_deg, mean_length


def _compute_moran_i_fast(
    z: np.ndarray,
    W: np.ndarray,
    S0: float,
    n: int
) -> float:
    """Fast internal Moran's I calculator using pre-built weight matrix."""
    sum_z2 = np.sum(z ** 2)
    if sum_z2 == 0:
        return 0.0
    numerator = float(z @ W @ z)
    return (n / S0) * (numerator / sum_z2)


def run_sensitivity_simulation(
    values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    n_simulations: int = 999,
    seed: int = 42
) -> dict:
    """Monte Carlo simulation for Global Moran's I sensitivity assessment.

    Returns:
        A dictionary with observed_i, simulated_mean, simulated_std,
        empirical_p, percentile_5, percentile_95, and simulated_values.
    """
    n = len(id_order)
    if n < 4:
        raise ValueError("Sensitivity simulation requires at least 4 features.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    # Build dense weight matrix
    W = np.zeros((n, n))
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                W[i, j] = w
    np.fill_diagonal(W, 0.0)
    S0 = float(np.sum(W))

    if S0 == 0:
        raise ValueError("No spatial neighbors found. Cannot run simulation.")

    # Observed Moran's I
    y = np.array([float(values[i]) for i in range(n)])
    z_obs = y - np.mean(y)
    observed_i = _compute_moran_i_fast(z_obs, W, S0, n)

    # Monte Carlo permutations
    rng = np.random.default_rng(seed)
    sim_values = np.zeros(n_simulations)

    for s in range(n_simulations):
        y_perm = rng.permutation(y)
        z_perm = y_perm - np.mean(y_perm)
        sim_values[s] = _compute_moran_i_fast(z_perm, W, S0, n)

    # Empirical p-value (two-tailed)
    count_extreme = np.sum(np.abs(sim_values) >= abs(observed_i))
    empirical_p = float((count_extreme + 1) / (n_simulations + 1))

    return {
        "observed_i": float(observed_i),
        "simulated_mean": float(np.mean(sim_values)),
        "simulated_std": float(np.std(sim_values)),
        "empirical_p": empirical_p,
        "percentile_5": float(np.percentile(sim_values, 5)),
        "percentile_95": float(np.percentile(sim_values, 95)),
        "simulated_values": sim_values.tolist()
    }


def calculate_incremental_autocorrelation(
    x: np.ndarray,
    y_coords: np.ndarray,
    values: np.ndarray,
    start_dist: float,
    dist_increment: float,
    n_increments: int
) -> list[dict]:
    """Calculates Global Moran's I at multiple distance bands.

    Returns:
        A list of dicts, each with keys: distance, morans_i, expected_i, z_score, p_value.
    """
    n = len(x)
    if n < 4:
        raise ValueError("Incremental autocorrelation requires at least 4 features.")

    coords = np.column_stack((x, y_coords))
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    y_mean = np.mean(values)
    z = values - y_mean
    sum_z2 = np.sum(z ** 2)

    if sum_z2 == 0:
        return [{"distance": start_dist + i * dist_increment,
                 "morans_i": 0.0, "expected_i": -1.0 / (n - 1),
                 "z_score": 0.0, "p_value": 1.0}
                for i in range(n_increments)]

    results = []
    for inc in range(n_increments):
        threshold = start_dist + inc * dist_increment

        W = (dists_matrix <= threshold).astype(float)
        np.fill_diagonal(W, 0.0)

        S0 = np.sum(W)
        if S0 == 0:
            results.append({
                "distance": threshold,
                "morans_i": 0.0,
                "expected_i": -1.0 / (n - 1),
                "z_score": 0.0,
                "p_value": 1.0
            })
            continue

        numerator = float(z @ W @ z)
        morans_i = (n / S0) * (numerator / sum_z2)
        expected_i = -1.0 / (n - 1)

        # Variance under randomization (simplified)
        S1 = float(np.sum((W + W.T) ** 2)) / 2.0
        row_sums = np.sum(W, axis=1)
        col_sums = np.sum(W, axis=0)
        S2 = float(np.sum((row_sums + col_sums) ** 2))

        sum_z4 = np.sum(z ** 4)
        D = (n * sum_z4) / (sum_z2 ** 2)

        num_var = (n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
                   - D * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2))
        den_var = (n - 1) * (n - 2) * (n - 3) * S0 ** 2

        variance = num_var / den_var - (expected_i ** 2) if den_var > 0 else 0.0

        if variance > 0:
            z_score = (morans_i - expected_i) / math.sqrt(variance)
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
        else:
            z_score = 0.0
            p_value = 1.0

        results.append({
            "distance": threshold,
            "morans_i": float(morans_i),
            "expected_i": float(expected_i),
            "z_score": float(z_score),
            "p_value": float(p_value)
        })

    return results


def calculate_exploratory_regression(
    y: np.ndarray,
    X_data: np.ndarray,
    x_names: list[str],
    max_vars: int | None = None
) -> list[dict]:
    """Tests all possible OLS variable combinations and returns ranked models.

    Returns:
        A sorted list of model dicts with keys: variables, r2, adj_r2, aic, coefficients.
    """
    from itertools import combinations

    n, total_p = X_data.shape
    if max_vars is None:
        max_vars = min(total_p, 5)  # cap at 5 to keep runtime manageable
    max_vars = min(max_vars, total_p)

    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return []

    models = []

    for k in range(1, max_vars + 1):
        for combo in combinations(range(total_p), k):
            X_sub = X_data[:, combo]
            X_design = np.column_stack((np.ones(n), X_sub))

            try:
                xtx_inv = np.linalg.pinv(X_design.T @ X_design)
                beta = xtx_inv @ X_design.T @ y
            except Exception:
                continue

            y_pred = X_design @ beta
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)

            p = len(combo)
            df = n - p - 1
            if df <= 0:
                continue

            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / df) if df > 0 else 0.0

            # AICc
            if ss_res > 0 and n > 0:
                log_lik = -n / 2.0 * (np.log(2.0 * np.pi * ss_res / n) + 1.0)
                k_params = p + 2  # intercept + vars + variance
                aic = -2.0 * log_lik + 2.0 * k_params
                if n - k_params - 1 > 0:
                    aicc = aic + (2.0 * k_params * (k_params + 1)) / (n - k_params - 1)
                else:
                    aicc = float('inf')
            else:
                aicc = float('inf')

            var_names = [x_names[i] for i in combo]
            coef_dict = {"Intercept": float(beta[0])}
            for idx, vi in enumerate(combo):
                coef_dict[x_names[vi]] = float(beta[idx + 1])

            models.append({
                "variables": var_names,
                "r2": float(r2),
                "adj_r2": float(adj_r2),
                "aicc": float(aicc),
                "coefficients": coef_dict,
                "n_vars": p
            })

    # Sort by AICc ascending (best first)
    models.sort(key=lambda m: m["aicc"])
    return models
