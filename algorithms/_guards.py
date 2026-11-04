# -*- coding: utf-8 -*-
"""Input guards shared by the spatial-statistics tools (v3.13).

Four things went wrong silently in 3.12 and every weights-based tool needs the
same answer to each of them, so they live here once:

* a NaN or infinite attribute passed ``float()`` and poisoned the statistic
  (``finite_value`` / ``read_finite_values`` skip and count it);
* Queen or Rook weights on a point layer gave every feature zero neighbours
  and the run reported "Random" (``check_weight_choice`` refuses it);
* a distance band on a geographic CRS was read in degrees, so the default
  1000 made every feature a neighbour of every other (refused the same way);
* an engine refusing its input (``core.stats_engines.InputError``, contract
  C1) surfaced as a traceback (``engine_errors`` turns it into a plain
  Processing error).

Class labels for the two "nothing was computed" states are also defined here,
so every cluster output and its renderer use the same words (contract C2).
"""
from __future__ import annotations

import math
from contextlib import contextmanager

from qgis.core import NULL, QgsProcessingException, QgsWkbTypes

try:  # contract C1 (GS-NUM); a stand-in keeps the tools importable before it lands
    from ..core.stats_engines import InputError
except ImportError:  # pragma: no cover - only before C1 is merged
    class InputError(ValueError):
        """Stand-in for core.stats_engines.InputError."""

NO_DATA = "No data"
NO_NEIGHBOURS = "No neighbours"

CONTIGUITY_ON_POINTS = (
    "Queen and Rook contiguity need polygons that share a boundary; points never do, so every "
    "feature would have no neighbours and nothing could be computed. Choose K-Nearest Neighbors "
    "or a Distance Band for a point layer."
)
DISTANCE_ON_GEOGRAPHIC = (
    "The layer uses a geographic CRS ({crs}), so a distance band of {band:g} would be read in "
    "degrees, not metres. Reproject the layer to a projected CRS first (Vector > Data Management "
    "Tools > Reproject Layer) or choose K-Nearest Neighbors."
)
NO_NEIGHBOUR_GRAPH = (
    "No feature has a neighbour among the features with usable values under this weights "
    "definition, so the statistic cannot be computed. Widen the neighbourhood (a larger K or "
    "distance band) or check that the polygons actually touch."
)


def finite_value(raw):
    """``(value, reason)``: a finite float and ``None``, or ``None`` and why not.

    ``reason`` is ``"missing"`` (NULL), ``"non_numeric"`` or ``"non_finite"``
    (NaN or infinity, which ``float()`` happily accepts).
    """
    if raw is None or raw == NULL or str(raw) == "NULL":
        return None, "missing"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, "non_numeric"
    if not math.isfinite(value):
        return None, "non_finite"
    return value, None


def strict_float(raw) -> float:
    """``float(raw)`` that also refuses NaN and infinity (raises ValueError)."""
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value {raw!r}")
    return value


def to_finite(raw):
    """The finite float behind ``raw``, or ``None`` (NULL, text, NaN, inf)."""
    return finite_value(raw)[0]


def empty_skip_counts() -> dict:
    return {"missing": 0, "non_numeric": 0, "non_finite": 0}


def read_finite_values(source, field_names, feedback=None):
    """``(values, counts)`` for features whose every listed field is finite.

    ``values`` maps feature id to a float (one field) or a list of floats
    (several); ``counts`` says how many features were skipped and why, so a
    report can state it and the output layer can mark them "No data".
    """
    single = isinstance(field_names, str)
    names = [field_names] if single else list(field_names)
    values = {}
    counts = empty_skip_counts()
    for feature in source.getFeatures():
        if feedback is not None and feedback.isCanceled():
            break
        row = []
        reason = None
        for name in names:
            value, reason = finite_value(feature.attribute(name))
            if reason:
                break
            row.append(value)
        if reason:
            counts[reason] += 1
            continue
        values[feature.id()] = row[0] if single else row
    return values, counts


def skipped_total(counts: dict) -> int:
    return int(sum(int(v) for v in counts.values()))


def skip_message(counts: dict) -> str:
    """One plain sentence, or "" when nothing was skipped."""
    total = skipped_total(counts)
    if not total:
        return ""
    parts = []
    if counts.get("missing"):
        parts.append(f"{counts['missing']} with no value (NULL)")
    if counts.get("non_numeric"):
        parts.append(f"{counts['non_numeric']} with a non-numeric value")
    if counts.get("non_finite"):
        parts.append(f"{counts['non_finite']} with NaN or infinity")
    return (
        f"{total} feature(s) were left out of the analysis: " + ", ".join(parts)
        + ". They are written as 'No data' and do not enter the neighbourhoods of other features."
    )


def push_skips(feedback, counts: dict) -> None:
    message = skip_message(counts)
    if message and feedback is not None:
        feedback.pushWarning(message)


def is_point_layer(source) -> bool:
    try:
        geometry_type = QgsWkbTypes.geometryType(source.wkbType())
    except (AttributeError, TypeError, RuntimeError):
        return False
    point = getattr(QgsWkbTypes, "PointGeometry", None)
    if point is None:  # pragma: no cover - QGIS 4 without the shim
        from qgis.core import Qgis
        point = Qgis.GeometryType.Point
    return geometry_type == point


def is_geographic(source) -> bool:
    try:
        return bool(source.sourceCrs().isGeographic())
    except (AttributeError, RuntimeError):
        return False


def check_weight_choice(source, weight_type: str, distance_band: float = 0.0) -> None:
    """Refuse a weights definition that cannot mean what the user expects."""
    code = str(weight_type).lower()
    if code in ("queen", "rook") and is_point_layer(source):
        raise QgsProcessingException(CONTIGUITY_ON_POINTS)
    if code == "distance" and is_geographic(source):
        check_distance_crs(source, distance_band)


def check_distance_crs(source, distance_band: float = 0.0) -> None:
    """Refuse any distance threshold on a geographic CRS (General G, ...)."""
    if is_geographic(source):
        try:
            crs = source.sourceCrs().authid() or "geographic"
        except (AttributeError, RuntimeError):
            crs = "geographic"
        raise QgsProcessingException(DISTANCE_ON_GEOGRAPHIC.format(crs=crs, band=float(distance_band or 0.0)))


def build_checked_weights(source, weight_type, k_neighbors=5, distance_band=1000.0, feedback=None, **kwargs):
    """``build_weights_matrix`` after ``check_weight_choice``."""
    from ..core.weights import build_weights_matrix

    check_weight_choice(source, weight_type, distance_band)
    return build_weights_matrix(
        source, weight_type, k_neighbors=k_neighbors, distance_band=distance_band, feedback=feedback, **kwargs
    )


def restrict_weights(neighbors, weights, valid_ids):
    """Neighbours and row-standardised weights among ``valid_ids`` only.

    A neighbour that was dropped (NULL, NaN) leaves the row; the remaining
    weights are rescaled to sum to one, keeping any distance decay.
    """
    from ..core.analysis_diagnostics import filter_weights_to_valid_ids

    return filter_weights_to_valid_ids(neighbors, valid_ids, weights=weights)


def neighbour_counts(neighbors, valid_ids) -> dict:
    valid = set(valid_ids)
    return {fid: sum(1 for nid in neighbors.get(fid, []) if nid in valid) for fid in valid_ids}


def require_neighbours(neighbors, valid_ids) -> None:
    """Refuse a graph in which no usable feature has a usable neighbour."""
    if valid_ids and not any(n > 0 for n in neighbour_counts(neighbors, valid_ids).values()):
        raise QgsProcessingException(NO_NEIGHBOUR_GRAPH)


@contextmanager
def engine_errors():
    """Turn an engine's ``InputError`` (a user sentence) into a Processing error."""
    try:
        yield
    except InputError as exc:
        raise QgsProcessingException(str(exc)) from exc


def set_fields_keeping_values(feature, fields) -> None:
    """``feature.setFields(fields)`` without losing the values it already has.

    PyQGIS's ``QgsFeature.setFields`` re-initialises the attributes (its
    ``initAttributes`` argument defaults to True in the bindings), so the
    3.12 pattern ``out = QgsFeature(f); out.setFields(out_fields)`` wrote
    every input attribute as NULL on QGIS 3.44 and 4.2.  Values are put back
    by position; the appended result fields start as NULL.
    """
    values = list(feature.attributes())
    feature.setFields(fields, True)
    count = fields.count()
    feature.setAttributes((values + [None] * count)[:count])


def stop_if_cancelled(feedback) -> None:
    """Raise rather than hand back a half-written layer that reads as complete."""
    if feedback is not None and feedback.isCanceled():
        raise QgsProcessingException("The run was cancelled; no output layer was produced.")
