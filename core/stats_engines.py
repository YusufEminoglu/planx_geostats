# -*- coding: utf-8 -*-
"""Spatial statistics engines for calculations."""
from __future__ import annotations

import logging
import math
import numpy as np

logger = logging.getLogger("PlanX-GeoStats")

# Try importing PySAL esda
HAS_ESDA = False
try:
    from esda.g import Gi_Local
    HAS_ESDA = True
except ImportError:
    logger.warning("esda is not available. Using native NumPy Getis-Ord Gi* engine.")


def calculate_getis_ord(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    star: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Getis-Ord Gi or Gi* statistics.

    Args:
        y: 1D NumPy array of values (ordered by id_order).
        neighbors: Dict mapping feature ID -> list of neighbor IDs.
        weights: Dict mapping feature ID -> list of weights.
        id_order: List of feature IDs corresponding to y.
        star: If True, calculates Gi* (with self loops). If False, calculates Gi.

    Returns:
        A tuple of:
          - z_scores: NumPy array of z-scores (floats).
          - p_values: NumPy array of p-values (floats).
          - conf_bins: NumPy array of confidence bins (-3 to 3) (ints).
    """
    n = len(y)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    conf_bins = np.zeros(n, dtype=int)

    if n <= 1:
        return z_scores, p_values, conf_bins

    # Check if we should use the PySAL implementation
    # We will use the NumPy fallback by default as it is extremely fast and robust,
    # and has zero external dependencies, but we can also use PySAL if preferred.
    # We implement the NumPy calculation directly because it is fully self-contained.
    
    # Calculate global parameters
    y_mean = np.mean(y)
    y_var = np.var(y)
    y_std = np.std(y)

    if y_std == 0:
        logger.warning("Standard deviation of the target field is zero. Gi* cannot be calculated.")
        return z_scores, p_values, conf_bins

    # Create mapping from feature ID to index in the array
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    # Build the weights matrix W
    # If star is True, we add the feature itself to its neighbors before row-standardization
    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        # Filter out invalid neighbor IDs
        valid_neigh_indices = []
        for nid in f_neighs:
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])

        # If star=True, add self-loop
        if star:
            if idx not in valid_neigh_indices:
                valid_neigh_indices.append(idx)

        num_neighbors = len(valid_neigh_indices)
        if num_neighbors == 0:
            z_scores[idx] = 0.0
            p_values[idx] = 1.0
            conf_bins[idx] = 0
            continue

        # Row-standardized weights
        w_row = np.ones(num_neighbors) / num_neighbors

        # Calculate values
        y_neigh = y[valid_neigh_indices]
        sum_w_x = np.sum(w_row * y_neigh)
        sum_w = np.sum(w_row)  # always 1.0 for row-standardized W, unless islands
        sum_w2 = np.sum(w_row ** 2)

        # Getis-Ord Gi* formula:
        # Numerator: sum_w_x - X_bar * sum_w
        # Denominator: S * sqrt( (n * sum_w2 - (sum_w)^2) / (n - 1) )
        numerator = sum_w_x - y_mean * sum_w
        
        denom_term = (n * sum_w2 - (sum_w ** 2)) / (n - 1)
        if denom_term < 0:
            denom_term = 0.0
            
        denominator = y_std * math.sqrt(denom_term)

        if denominator > 0:
            z = numerator / denominator
            z_scores[idx] = z
            
            # Two-tailed p-value using erf approximation
            p = 1.0 - math.erf(abs(z) / math.sqrt(2.0))
            p_values[idx] = p

            # Determine confidence bins (-3, -2, -1, 0, 1, 2, 3)
            # 99%: z > 2.58, p < 0.01
            # 95%: z > 1.96, p < 0.05
            # 90%: z > 1.65, p < 0.10
            if p < 0.01:
                if z > 0:
                    conf_bins[idx] = 3
                else:
                    conf_bins[idx] = -3
            elif p < 0.05:
                if z > 0:
                    conf_bins[idx] = 2
                else:
                    conf_bins[idx] = -2
            elif p < 0.10:
                if z > 0:
                    conf_bins[idx] = 1
                else:
                    conf_bins[idx] = -1
            else:
                conf_bins[idx] = 0
        else:
            z_scores[idx] = 0.0
            p_values[idx] = 1.0
            conf_bins[idx] = 0

    return z_scores, p_values, conf_bins
