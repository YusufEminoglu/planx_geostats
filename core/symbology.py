# -*- coding: utf-8 -*-
"""Shared automatic QGIS symbology helpers for PlanX GeoStats output layers.

Mirrors core/charts.py's "one small function per pattern, same call
convention everywhere" shape, but returns QGIS renderer/symbol objects
instead of HTML strings, since symbology must be applied directly to a
QgsVectorLayer in postProcessAlgorithm(). Extracted from the categorized
(LISA quadrant) and graduated (diverging std-dev) renderers several tools
already hand-rolled identically (alg_local_moran.py, alg_spatial_regression.py,
alg_skater.py, ...); new tools should call these instead of re-writing the
QgsSymbol/stroke boilerplate again."""
from __future__ import annotations

from qgis.PyQt.QtGui import QColor
from qgis.core import (
    NULL,
    QgsAggregateCalculator,
    QgsCategorizedSymbolRenderer,
    QgsGraduatedSymbolRenderer,
    QgsRendererCategory,
    QgsRendererRange,
    QgsSymbol,
)

STROKE_COLOR = "#b0b0b0"
STROKE_WIDTH = 0.1
FILL_OPACITY = 0.85

QUALITATIVE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

DIVERGING_RDBU_7 = [
    "#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#b2182b",
]

LISA_QUADRANT_STYLE = [
    ("HH", "#e31a1c", "High-High (HH)"),
    ("LL", "#1f78b4", "Low-Low (LL)"),
    ("HL", "#fb9a99", "High-Low (HL) outlier"),
    ("LH", "#a6cee3", "Low-High (LH) outlier"),
    ("Not Significant", "#f7f7f7", "Not Significant"),
]

_SIGMA_BREAKS = [-9999.0, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 9999.0]
_SIGMA_LABELS = [
    "< -2.5 Std Dev (Underprediction)",
    "-2.5 to -1.5 Std Dev",
    "-1.5 to -0.5 Std Dev",
    "-0.5 to 0.5 Std Dev (Near Zero)",
    "0.5 to 1.5 Std Dev",
    "1.5 to 2.5 Std Dev",
    "> 2.5 Std Dev (Overprediction)",
]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _interp_hex(hex_light: str, hex_dark: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    r0, g0, b0 = _hex_to_rgb(hex_light)
    r1, g1, b1 = _hex_to_rgb(hex_dark)
    r = round(r0 + (r1 - r0) * t)
    g = round(g0 + (g1 - g0) * t)
    b = round(b0 + (b1 - b0) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def styled_symbol(geometry_type, color_hex: str, opacity: float = FILL_OPACITY):
    """One QgsSymbol with the consistent fill/stroke styling every renderer
    in this plugin already uses, so every output layer looks like it belongs
    to the same family regardless of which tool produced it."""
    symbol = QgsSymbol.defaultSymbol(geometry_type)
    symbol.setColor(QColor(color_hex))
    symbol.setOpacity(opacity)
    if symbol.symbolLayerCount() > 0:
        sl = symbol.symbolLayer(0)
        if hasattr(sl, "setStrokeColor"):
            sl.setStrokeColor(QColor(STROKE_COLOR))
        if hasattr(sl, "setStrokeWidth"):
            sl.setStrokeWidth(STROKE_WIDTH)
        if hasattr(sl, "setOutlineColor"):
            sl.setOutlineColor(QColor(STROKE_COLOR))
    return symbol


def categorized_renderer(geometry_type, field_name: str, category_defs):
    """category_defs: iterable of (value, color_hex, label)."""
    categories = [
        QgsRendererCategory(value, styled_symbol(geometry_type, color_hex), label, True)
        for value, color_hex, label in category_defs
    ]
    return QgsCategorizedSymbolRenderer(field_name, categories)


def lisa_quadrant_renderer(geometry_type, field_name: str = "quadrant"):
    """The HH/LL/HL/LH/Not Significant categorized renderer every LISA-family
    tool (Local Moran's I, Getis-Ord Gi*, Local Geary's C, Bivariate LISA,
    Bivariate Lee's L) already applies."""
    return categorized_renderer(geometry_type, field_name, LISA_QUADRANT_STYLE)


def diverging_std_dev_renderer(geometry_type, field_name: str):
    """7-class diverging std-dev renderer for a field that is ALREADY a
    standardized residual (e.g. std_res, mgwr_std). Fixed sigma breakpoints,
    identical to the convention alg_spatial_regression.py established."""
    ranges = [
        QgsRendererRange(_SIGMA_BREAKS[i], _SIGMA_BREAKS[i + 1], styled_symbol(geometry_type, DIVERGING_RDBU_7[i]), _SIGMA_LABELS[i])
        for i in range(7)
    ]
    return QgsGraduatedSymbolRenderer(field_name, ranges)


def diverging_residual_renderer(layer, geometry_type, field_name: str):
    """Same 7-class diverging convention, but data-driven: computes mean and
    std dev of a RAW (non-standardized) residual field directly off the
    built layer via QGIS's own aggregate functions, so tools that only write
    a plain '<engine>_resid' column (most of the ML regressors) get the same
    residual-diagnostic map every spatial-econometric tool already has,
    without needing a separate standardized-residual output field. Returns
    None if the layer is unavailable, the field is missing, has fewer than 2
    values, or has zero variance (nothing meaningful to diverge around)."""
    if layer is None:
        return None
    field_idx = layer.fields().lookupField(field_name)
    if field_idx < 0:
        return None
    mean, mean_ok = layer.aggregate(QgsAggregateCalculator.Mean, field_name)
    std, std_ok = layer.aggregate(QgsAggregateCalculator.StDev, field_name)
    if not mean_ok or not std_ok or not std:
        return None
    ranges = []
    for i in range(7):
        lo_sigma, hi_sigma = _SIGMA_BREAKS[i], _SIGMA_BREAKS[i + 1]
        lo = mean + lo_sigma * std if lo_sigma > -9999.0 else -1e15
        hi = mean + hi_sigma * std if hi_sigma < 9999.0 else 1e15
        ranges.append(QgsRendererRange(lo, hi, styled_symbol(geometry_type, DIVERGING_RDBU_7[i]), _SIGMA_LABELS[i]))
    return QgsGraduatedSymbolRenderer(field_name, ranges)


def sequential_quantile_renderer(
    layer,
    geometry_type,
    field_name: str,
    n_classes: int = 5,
    ramp_light: str = "#eff6ff",
    ramp_dark: str = "#1e3a8a",
):
    """Data-driven equal-count (quantile) graduated renderer for a
    confidence/uncertainty field (e.g. rf_conf, unc_std, conf_width) - light
    to dark sequential single-hue, colorblind-safe, matching the same
    sequential ramp core/charts.py::heatmap_table_svg() uses for confusion
    matrices. Reads values directly off the built layer."""
    if layer is None:
        return None
    field_idx = layer.fields().lookupField(field_name)
    if field_idx < 0:
        return None
    values = sorted(
        float(v) for v in layer.uniqueValues(field_idx)
        if v is not None and v != NULL
    )
    n = len(values)
    if n < 2:
        return None
    ranges = []
    for i in range(n_classes):
        lo_idx = int(i * n / n_classes)
        hi_idx = min(int((i + 1) * n / n_classes), n - 1)
        lo, hi = values[lo_idx], values[hi_idx]
        if i == n_classes - 1:
            hi = values[-1] + max(abs(values[-1]) * 1e-9, 1e-9)
        t = i / max(n_classes - 1, 1)
        color_hex = _interp_hex(ramp_light, ramp_dark, t)
        label = f"{lo:.4g} - {hi:.4g}"
        ranges.append(QgsRendererRange(lo, hi, styled_symbol(geometry_type, color_hex), label))
    return QgsGraduatedSymbolRenderer(field_name, ranges)


def categorical_id_renderer(
    layer,
    geometry_type,
    field_name: str,
    palette=None,
    noise_value: int | None = None,
    noise_color: str = "#cccccc",
    noise_label: str = "Noise / unassigned",
):
    """Data-driven qualitative renderer for a cluster-id-style integer field
    (DBSCAN/HDBSCAN/GMM cluster_id, cv_fold), cycling through a fixed
    10-color qualitative palette - the same convention alg_skater.py already
    established for region_id. noise_value (e.g. -1 for DBSCAN/HDBSCAN noise
    points) always renders in a fixed neutral gray regardless of palette
    cycling, so noise reads consistently across every clustering tool."""
    if layer is None:
        return None
    palette = palette or QUALITATIVE_PALETTE
    field_idx = layer.fields().lookupField(field_name)
    if field_idx < 0:
        return None
    unique_vals = sorted(
        v for v in layer.uniqueValues(field_idx)
        if v is not None and v != NULL
    )
    if not unique_vals:
        return None
    categories = []
    for val in unique_vals:
        int_val = int(val)
        if noise_value is not None and int_val == noise_value:
            categories.append(QgsRendererCategory(val, styled_symbol(geometry_type, noise_color), noise_label, True))
            continue
        color_hex = palette[int_val % len(palette)]
        categories.append(QgsRendererCategory(val, styled_symbol(geometry_type, color_hex), f"Cluster {int_val}", True))
    return QgsCategorizedSymbolRenderer(field_name, categories)


def apply_renderer(layer, renderer) -> bool:
    """Set the renderer and trigger a repaint, or no-op safely if renderer
    construction returned None (missing field / degenerate data)."""
    if renderer is None or layer is None:
        return False
    layer.setRenderer(renderer)
    layer.triggerRepaint()
    return True
