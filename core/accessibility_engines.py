# -*- coding: utf-8 -*-
"""Straight-line (Euclidean) accessibility metrics for the PlanX GeoStats Lab
Accessibility group: 2SFCA, gravity-based accessibility, and nearest-facility
coverage gap.

These use straight-line distance between feature centroids, not network
routing distance - a standard simplification when a routable street network
is not available, and the one every tool in this group states explicitly in
its help text. For a network-distance version of the same idea, see Network
Reach in the Network Centrality and Space Syntax group (07) on a street line
layer covering the same study area.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .weights import geometry_centroid_point


def extract_coords_and_value(source, value_field: Optional[str] = None, feedback=None) -> dict:
    """Extract centroid (x, y) per feature, plus an optional numeric field's
    value (defaults to 1.0 per feature when value_field is None - useful for
    demand layers where every record counts as one unit)."""
    fields = source.fields()
    value_idx = fields.lookupField(value_field) if value_field else -1
    if value_field and value_idx < 0:
        raise ValueError(f"Field '{value_field}' not found.")

    coords, values, valid_fids = [], [], []
    skipped = 0
    total = source.featureCount() or 1
    for idx, feature in enumerate(source.getFeatures()):
        if feedback and feedback.isCanceled():
            break
        centroid = geometry_centroid_point(feature.geometry())
        if centroid is None:
            skipped += 1
            continue
        if value_idx >= 0:
            raw = feature.attribute(value_idx)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not np.isfinite(value):
                skipped += 1
                continue
        else:
            value = 1.0
        coords.append([centroid.x(), centroid.y()])
        values.append(value)
        valid_fids.append(feature.id())
        if feedback:
            feedback.setProgress(int(30 * (idx / total)))

    return {
        "coords": np.array(coords, dtype=float),
        "values": np.array(values, dtype=float),
        "valid_fids": valid_fids,
        "skipped": skipped,
    }


def _pairwise_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def two_step_fca(
    demand_coords: np.ndarray, demand_values: np.ndarray, supply_coords: np.ndarray,
    supply_values: np.ndarray, threshold: float,
) -> dict:
    """Classic (binary-catchment) 2SFCA. Step 1: each supply point's
    provider-to-population ratio over the demand within its catchment. Step
    2: each demand point's accessibility = sum of ratios of every supply
    point within its own catchment."""
    dist = _pairwise_distance(demand_coords, supply_coords)
    within = dist <= threshold
    demand_sum_per_supply = within.T @ demand_values
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(demand_sum_per_supply > 0, supply_values / demand_sum_per_supply, 0.0)
    accessibility = within @ ratio
    return {"accessibility": accessibility, "ratio_per_supply": ratio, "demand_sum_per_supply": demand_sum_per_supply}


def gravity_accessibility(
    demand_coords: np.ndarray, supply_coords: np.ndarray, supply_values: np.ndarray,
    decay: str = "gaussian", decay_param: float = 500.0,
) -> dict:
    """Gravity-model accessibility: each demand point's score is the sum of
    every supply point's capacity weighted by a distance-decay function of
    the straight-line distance between them (no hard cutoff, unlike 2SFCA)."""
    dist = _pairwise_distance(demand_coords, supply_coords)
    if decay == "power":
        with np.errstate(divide="ignore"):
            weight = np.where(dist > 0, dist ** (-decay_param), 1.0)
    elif decay == "exponential":
        weight = np.exp(-dist / max(decay_param, 1e-9))
    elif decay == "gaussian":
        weight = np.exp(-(dist ** 2) / (2 * (max(decay_param, 1e-9) ** 2)))
    else:
        raise ValueError(f"Unknown decay function: {decay}")
    accessibility = weight @ supply_values
    return {"accessibility": accessibility}


def nearest_facility_gap(demand_coords: np.ndarray, supply_coords: np.ndarray, threshold: float) -> dict:
    dist = _pairwise_distance(demand_coords, supply_coords)
    nearest_dist = dist.min(axis=1)
    nearest_idx = dist.argmin(axis=1)
    covered = nearest_dist <= threshold
    return {"nearest_dist": nearest_dist, "nearest_idx": nearest_idx, "covered": covered}
