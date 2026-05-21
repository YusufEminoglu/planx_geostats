# -*- coding: utf-8 -*-
"""Spatial weight matrix generation from QGIS layers."""
from __future__ import annotations

import logging
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


def build_weights_matrix(
    layer: QgsVectorLayer,
    weight_type: str,
    k_neighbors: int = 5,
    distance_band: float = 1000.0,
    feedback: Optional[QgsFeedback] = None
) -> tuple[dict[int, list[int]], dict[int, list[float]], list[int], Optional[object]]:
    """Builds a spatial weights matrix from a QGIS vector layer.

    Args:
        layer: The input QGIS vector layer.
        weight_type: One of 'queen', 'rook', 'knn', 'distance'.
        k_neighbors: Number of neighbors for KNN weights.
        distance_band: Distance threshold for distance-band weights.
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

        if geom.isEmpty():
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
                # If they share at least one vertex or edge
                if geom.intersects(features_dict[cid].geometry()):
                    f_neighs.append(cid)

        elif weight_type.lower() == 'rook':
            # Intersects bounding box query
            bbox = geom.boundingBox()
            candidates = spatial_index.intersects(bbox)
            for cid in candidates:
                if cid == fid:
                    continue
                # Check actual intersection geometry
                inter = geom.intersection(features_dict[cid].geometry())
                # Dimension >= 1 means they share a boundary line/curve, not just a vertex
                if not inter.isEmpty() and inter.dimension() >= 1:
                    f_neighs.append(cid)

        elif weight_type.lower() == 'knn':
            centroid = geom.centroid().asPoint()
            # Query k + 1 because QGIS nearest-neighbor results include the feature itself.
            nearest = nearest_neighbor_ids(spatial_index, centroid, k_neighbors + 1)
            f_neighs = [nid for nid in nearest if nid != fid][:k_neighbors]

        elif weight_type.lower() == 'distance':
            centroid = geom.centroid().asPoint()
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
                other_centroid = features_dict[cid].geometry().centroid().asPoint()
                dist = centroid.distance(other_centroid)
                if dist <= distance_band:
                    f_neighs.append(cid)

        neighbors[fid] = f_neighs

        if feedback and i % max(1, total // 20) == 0:
            feedback.setProgress(int(50 * (i / total)))  # 0 to 50% for weights generation

    # Row-standardize weights
    for fid in id_order:
        neighs = neighbors.get(fid, [])
        n = len(neighs)
        if n > 0:
            weights[fid] = [1.0 / n] * n
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
