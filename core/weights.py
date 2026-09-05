# -*- coding: utf-8 -*-
"""Spatial weight matrix generation from QGIS layers."""
from __future__ import annotations

import logging
import csv
from typing import Optional

from qgis.core import (
    QgsFeature,
    QgsVectorLayer,
    QgsSpatialIndex,
    QgsRectangle,
    QgsFeedback
)

logger = logging.getLogger("PlanX GeoStats Lab")

# Try importing libpysal
HAS_PYSAL = False
try:
    import libpysal
    HAS_PYSAL = True
except ImportError:
    logger.warning("libpysal is not available. Using native fallback weights.")


def nearest_neighbor_ids(spatial_index, point, count: int) -> list[int]:
    """Return nearest feature ids across QGIS API naming variants."""
    if hasattr(spatial_index, "nearestNeighbor"):
        return list(spatial_index.nearestNeighbor(point, count))
    return list(spatial_index.nearestNeighbors(point, count))


def geometry_is_missing_or_empty(geometry) -> bool:
    """Return True for absent or empty QGIS geometries across provider variants."""
    try:
        return geometry is None or geometry.isEmpty()
    except (AttributeError, RuntimeError, TypeError):
        return True


def geometry_centroid_point(geometry):
    """Return a geometry centroid point, or None when a provider cannot produce one."""
    if geometry_is_missing_or_empty(geometry):
        return None
    try:
        centroid = geometry.centroid()
        if geometry_is_missing_or_empty(centroid):
            return None
        return centroid.asPoint()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def build_weights_matrix(
    layer: QgsVectorLayer,
    weight_type: str,
    k_neighbors: int = 5,
    distance_band: float = 1000.0,
    neighbor_order: int = 1,
    repair_islands: bool = False,
    distance_decay_power: float = 0.0,
    feedback: Optional[QgsFeedback] = None
) -> tuple[dict[int, list[int]], dict[int, list[float]], list[int], Optional[object]]:
    """Builds a spatial weights matrix from a QGIS vector layer.

    Args:
        layer: The input QGIS vector layer.
        weight_type: One of 'queen', 'rook', 'knn', 'distance'.
        k_neighbors: Number of neighbors for KNN weights.
        distance_band: Distance threshold for distance-band weights.
        neighbor_order: Graph order; 2 includes neighbors-of-neighbors, etc.
        repair_islands: Connect zero-neighbor features to their nearest feature.
        distance_decay_power: If positive, use inverse-distance weights.
        feedback: QgsFeedback for cancellation and progress reporting.

    Returns:
        A tuple of:
          - neighbors: Dict mapping feature ID (int) -> list of neighbor IDs (ints)
          - weights: Dict mapping feature ID (int) -> list of row-standardized weights (floats)
          - id_order: List of feature IDs in the order they were processed
          - pysal_w: libpysal.weights.W object, or None if PySAL is not available
    """
    neighbors: dict[int, list[int]] = {}
    weights: dict[int, list[float]] = {}
    id_order: list[int] = []

    # Get all features and store in a dictionary for quick lookup by ID
    features_dict: dict[int, QgsFeature] = {}
    for f in layer.getFeatures():
        if feedback and feedback.isCanceled():
            return {}, {}, [], None
        features_dict[f.id()] = QgsFeature(f)
        id_order.append(f.id())

    total = len(id_order)
    if total == 0:
        return {}, {}, [], None

    # Construct the native spatial index
    spatial_index = QgsSpatialIndex(layer.getFeatures())

    # Build neighbors
    for i, fid in enumerate(id_order):
        if feedback and feedback.isCanceled():
            return {}, {}, [], None

        feature = features_dict[fid]
        geom = feature.geometry()

        if geometry_is_missing_or_empty(geom):
            neighbors[fid] = []
            continue

        f_neighs: list[int] = []

        if weight_type.lower() == 'queen':
            # Intersects bounding box query
            bbox = geom.boundingBox()
            candidates = spatial_index.intersects(bbox)
            for cid in candidates:
                if cid == fid:
                    continue
                candidate_geom = features_dict[cid].geometry()
                if geometry_is_missing_or_empty(candidate_geom):
                    continue
                # If they share at least one vertex or edge
                if geom.intersects(candidate_geom):
                    f_neighs.append(cid)

        elif weight_type.lower() == 'rook':
            # Intersects bounding box query
            bbox = geom.boundingBox()
            candidates = spatial_index.intersects(bbox)
            for cid in candidates:
                if cid == fid:
                    continue
                candidate_geom = features_dict[cid].geometry()
                if geometry_is_missing_or_empty(candidate_geom):
                    continue
                # Check actual intersection geometry
                inter = geom.intersection(candidate_geom)
                # A rook neighbor shares a boundary segment, not merely a
                # corner. QgsGeometry.dimension() was removed in QGIS 4;
                # positive intersection length is equivalent here and is
                # supported by both QGIS 3 and 4.
                if not geometry_is_missing_or_empty(inter) and inter.length() > 0.0:
                    f_neighs.append(cid)

        elif weight_type.lower() == 'knn':
            centroid = geometry_centroid_point(geom)
            if centroid is None:
                neighbors[fid] = []
                continue
            # Query k + 1 because QGIS nearest-neighbor results include the feature itself.
            nearest = nearest_neighbor_ids(spatial_index, centroid, k_neighbors + 1)
            f_neighs = [nid for nid in nearest if nid != fid][:k_neighbors]

        elif weight_type.lower() == 'distance':
            centroid = geometry_centroid_point(geom)
            if centroid is None:
                neighbors[fid] = []
                continue
            # Bounding box of size 2D x 2D around the centroid
            bbox = QgsRectangle(
                centroid.x() - distance_band,
                centroid.y() - distance_band,
                centroid.x() + distance_band,
                centroid.y() + distance_band
            )
            candidates = spatial_index.intersects(bbox)
            for cid in candidates:
                if cid == fid:
                    continue
                candidate_geom = features_dict[cid].geometry()
                if geometry_is_missing_or_empty(candidate_geom):
                    continue
                other_centroid = geometry_centroid_point(candidate_geom)
                if other_centroid is None:
                    continue
                dist = centroid.distance(other_centroid)
                if dist <= distance_band:
                    f_neighs.append(cid)

        neighbors[fid] = f_neighs

        if feedback and i % max(1, total // 20) == 0:
            feedback.setProgress(int(50 * (i / total)))  # 0 to 50% for weights generation

    if repair_islands and total > 1:
        for fid in id_order:
            if neighbors.get(fid):
                continue
            centroid = geometry_centroid_point(features_dict[fid].geometry())
            if centroid is None:
                continue
            nearest = nearest_neighbor_ids(spatial_index, centroid, 2)
            replacement = next((nid for nid in nearest if nid != fid), None)
            if replacement is not None:
                neighbors[fid] = [replacement]
                if fid not in neighbors.get(replacement, []):
                    neighbors.setdefault(replacement, []).append(fid)

    if neighbor_order > 1:
        neighbors = expand_higher_order_neighbors(neighbors, neighbor_order)

    centroids = {
        fid: geometry_centroid_point(features_dict[fid].geometry())
        for fid in id_order
    }

    # Row-standardize binary or inverse-distance weights.
    for fid in id_order:
        neighs = neighbors.get(fid, [])
        if neighs and distance_decay_power > 0 and centroids.get(fid) is not None:
            raw = []
            for neighbor_id in neighs:
                other = centroids.get(neighbor_id)
                distance = centroids[fid].distance(other) if other is not None else 0.0
                raw.append(1.0 / max(distance, 1e-12) ** distance_decay_power)
            total_weight = sum(raw)
            weights[fid] = [value / total_weight for value in raw]
        elif neighs:
            weights[fid] = [1.0 / len(neighs)] * len(neighs)
        else:
            weights[fid] = []

    # Wrap in PySAL W if available
    pysal_w = None
    if HAS_PYSAL:
        try:
            # PySAL expects dictionary keys and values to match ID type
            pysal_w = libpysal.weights.W(neighbors, weights, id_order=id_order)
        except Exception as e:
            logger.error("Failed to build libpysal weight matrix: %s", e)

    return neighbors, weights, id_order, pysal_w


def expand_higher_order_neighbors(
    neighbors: dict[int, list[int]],
    order: int,
    include_lower: bool = True,
) -> dict[int, list[int]]:
    """Expand an adjacency graph to the requested shortest-path order."""
    if order <= 1:
        return {fid: list(dict.fromkeys(values)) for fid, values in neighbors.items()}
    expanded = {}
    for fid in neighbors:
        visited = {fid}
        frontier = {fid}
        collected = set()
        for depth in range(1, order + 1):
            next_frontier = {
                neighbor
                for current in frontier
                for neighbor in neighbors.get(current, [])
                if neighbor not in visited
            }
            visited.update(next_frontier)
            if include_lower or depth == order:
                collected.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        expanded[fid] = sorted(collected)
    return expanded


def weights_diagnostics(neighbors: dict[int, list[int]]) -> dict:
    """Compute graph connectivity, cardinality, symmetry, and island metrics."""
    ids = list(neighbors)
    counts = [len(neighbors.get(fid, [])) for fid in ids]
    islands = [fid for fid in ids if not neighbors.get(fid)]
    asymmetric = sum(
        1 for fid in ids for neighbor in neighbors.get(fid, [])
        if fid not in neighbors.get(neighbor, [])
    )
    components = []
    unseen = set(ids)
    reverse = {fid: set(values) for fid, values in neighbors.items()}
    for fid, values in neighbors.items():
        for neighbor in values:
            reverse.setdefault(neighbor, set()).add(fid)
    while unseen:
        seed = min(unseen)
        component = set()
        frontier = {seed}
        while frontier:
            current = frontier.pop()
            if current in component:
                continue
            component.add(current)
            frontier.update(reverse.get(current, set()) - component)
        unseen.difference_update(component)
        components.append(component)
    return {
        "feature_count": len(ids),
        "edge_count": sum(counts),
        "island_count": len(islands),
        "island_ids": islands,
        "component_count": len(components),
        "largest_component": max((len(value) for value in components), default=0),
        "min_neighbors": min(counts, default=0),
        "mean_neighbors": sum(counts) / len(counts) if counts else 0.0,
        "max_neighbors": max(counts, default=0),
        "asymmetric_links": asymmetric,
    }


def export_weights_csv(path: str, neighbors, weights) -> None:
    """Write a portable source_id,target_id,weight adjacency list."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_id", "target_id", "weight"])
        for source_id in sorted(neighbors):
            for target_id, weight in zip(neighbors[source_id], weights.get(source_id, [])):
                writer.writerow([source_id, target_id, f"{float(weight):.17g}"])


def import_weights_csv(path: str, valid_ids=None):
    """Read a portable adjacency list and row-standardize its weights."""
    allowed = set(valid_ids) if valid_ids is not None else None
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source_id = int(row["source_id"])
            target_id = int(row["target_id"])
            if allowed is not None and (source_id not in allowed or target_id not in allowed):
                continue
            rows.setdefault(source_id, []).append((target_id, float(row.get("weight") or 1.0)))
    neighbors = {fid: [target for target, _ in values] for fid, values in rows.items()}
    weights = {}
    for fid, values in rows.items():
        total_weight = sum(max(0.0, weight) for _, weight in values)
        weights[fid] = [
            max(0.0, weight) / total_weight if total_weight else 1.0 / len(values)
            for _, weight in values
        ]
    if allowed is not None:
        for fid in allowed:
            neighbors.setdefault(fid, [])
            weights.setdefault(fid, [])
    return neighbors, weights
