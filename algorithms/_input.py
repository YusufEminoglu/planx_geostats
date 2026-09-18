# -*- coding: utf-8 -*-
"""Shared layer-reading helpers for PlanX GeoStats Lab algorithms.

The interpolation tools all start the same way: pull a numeric field off a
vector layer, take each feature's centroid as its observation location, drop
anything that is not a finite number, and hand the result to the same
diagnostics.  It lives here rather than in ``core`` because ``core`` is
deliberately numpy-only - the engines under it are tested without QGIS, which
is only possible while nothing under ``core`` imports it.

The return shape is the contract: ``(coords, values, summary, crs_warning)``,
or ``None`` when the user cancelled.  Callers pass the summary straight to
``push_diagnostics`` and to their report's diagnostics section.
"""
from __future__ import annotations

import numpy as np

from qgis.core import NULL, QgsProcessingException

from ..core.analysis_diagnostics import crs_unit_warning, numeric_quality_summary
from ..core.weights import geometry_centroid_point


def read_point_values(source, field_name, feedback):
    """Validated observation coordinates and values from a vector layer.

    Any geometry type is accepted and reduced to its centroid, so a polygon
    layer of districts can be interpolated as readily as a point layer - the
    tool says so in its report, because the centroid of a large polygon is a
    much weaker statement about where a value applies than a surveyed point.

    Features whose value is null, non-numeric or non-finite are dropped rather
    than treated as zero, and features with no usable centroid are counted
    separately and warned about, so the observation count in the report always
    matches the number of points that actually entered the fit.
    """
    field_idx = source.fields().lookupField(field_name)
    if field_idx < 0:
        raise QgsProcessingException(f"Target field '{field_name}' not found.")

    coords = []
    values = []
    value_by_id = {}
    missing_geometry = 0
    for feature in source.getFeatures():
        if feedback.isCanceled():
            return None
        raw = feature.attribute(field_name)
        if raw is None or raw == NULL or str(raw) == "NULL":
            continue
        try:
            z = float(raw)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(z):
            continue
        centroid = geometry_centroid_point(feature.geometry())
        if centroid is None:
            missing_geometry += 1
            continue
        coords.append([centroid.x(), centroid.y()])
        values.append(z)
        value_by_id[feature.id()] = z

    summary = numeric_quality_summary(source.featureCount(), value_by_id, values)
    warning = crs_unit_warning(source)
    if missing_geometry:
        feedback.pushWarning(
            f"{missing_geometry} feature(s) had no usable centroid and were "
            "excluded from the interpolation."
        )
    return coords, values, summary, warning


def require_non_constant(summary):
    """Refuse a field with no variation.

    A constant field is not an interpolation problem: every method would return
    that constant, and a cross-validation score of zero would describe the
    field rather than the method.  Refusing it here keeps that from being
    mistaken for a good result.
    """
    if summary["is_constant"]:
        raise QgsProcessingException(
            "The target field is constant, so every prediction would return "
            "that same value wherever it was made. Cross-validation would "
            "score zero without any method having done anything, which is why "
            "this is refused rather than reported as a perfect fit. There is "
            "nothing here to interpolate."
        )
