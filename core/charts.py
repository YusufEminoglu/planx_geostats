# -*- coding: utf-8 -*-
"""Dependency-free inline-SVG chart primitives for PlanX GeoStats Lab HTML
reports. Stdlib only (html, math) -- no numpy/matplotlib/PIL, no network
references -- so every chart renders identically offline inside QGIS's
embedded HTML viewer and clears the Hub security scan. Mirrors
core.reporting's fragment pattern: each function returns a self-contained
HTML/SVG string; chart_css() is the sibling CSS fragment callers inline once
per report, alongside analyst_guidance_css()."""
from __future__ import annotations

import html
import math

CHART_WIDTH = 640
CHART_HEIGHT = 320
MARGIN = {"top": 20, "right": 24, "bottom": 44, "left": 56}

COLOR_PRIMARY = "#4299e1"
COLOR_SECONDARY = "#f6ad55"
COLOR_TREND = "#e31a1c"
COLOR_GRID = "#cbd5e1"
COLOR_AXIS_TEXT = "#334155"
COLOR_QUADRANT_POS = "#dbeafe"
COLOR_QUADRANT_NEG = "#fff7ed"
HEATMAP_LOW = (239, 246, 255)
HEATMAP_HIGH = (30, 58, 138)

SERIES_COLORS = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TREND]


def chart_css() -> str:
    """Return CSS used by every chart_*.py-generated fragment below."""
    return """
.chart-block { margin: 18px 0; }
.chart-block h3 { margin: 0 0 8px; font-size: .95rem; color: #1f2937; }
.chart-block svg { display: block; }
.chart-caption { margin: 6px 0 0; font-size: .78rem; color: #64748b; }
.kpi-row { display: flex; flex-wrap: wrap; gap: 12px; margin: 4px 0 20px; }
.kpi-card { flex: 1 1 140px; background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px 14px; }
.kpi-card .kpi-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; color: #64748b; margin: 0 0 4px; }
.kpi-card .kpi-value { font-size: 1.35rem; font-weight: 700; color: #1a202c; font-family: monospace; margin: 0; }
.kpi-card .kpi-sublabel { font-size: .78rem; color: #64748b; margin: 4px 0 0; }
.kpi-card.tone-good { border-left: 4px solid #2f855a; }
.kpi-card.tone-warn { border-left: 4px solid #dd6b20; }
.kpi-card.tone-bad { border-left: 4px solid #c53030; }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _fmt(value: float) -> str:
    try:
        if value != value:  # NaN
            return "n/a"
    except TypeError:
        return _esc(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _svg_open(width: float, height: float, title: str = "") -> str:
    label = _esc(title) if title else "Chart"
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="xMidYMid meet" '
        f'role="img" aria-label="{label}" '
        f'style="width:100%;max-width:{width:.0f}px;height:auto;">'
    )


def _wrap_chart(title: str, body: str, caption: str = "") -> str:
    heading = f"<h3>{_esc(title)}</h3>" if title else ""
    cap = f'<p class="chart-caption">{_esc(caption)}</p>' if caption else ""
    return f'<div class="chart-block">{heading}{body}{cap}</div>'


def no_data_svg(
    width: int = CHART_WIDTH,
    height: int = CHART_HEIGHT,
    message: str = "No data available for this chart",
) -> str:
    return (
        f"{_svg_open(width, height, message)}"
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" stroke="{COLOR_GRID}"/>'
        f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
        f'fill="{COLOR_AXIS_TEXT}" font-size="12">{_esc(message)}</text>'
        f"</svg>"
    )


def _make_scale(domain: tuple[float, float], range_: tuple[float, float]):
    d0, d1 = domain
    r0, r1 = range_
    span = d1 - d0
    if span == 0:
        mid = (r0 + r1) / 2
        return lambda _v: mid

    def scale(v: float) -> float:
        return r0 + ((v - d0) / span) * (r1 - r0)

    return scale


def _axis_ticks(min_v: float, max_v: float, n: int = 5) -> list[float]:
    if min_v == max_v:
        return [min_v]
    step = (max_v - min_v) / max(n - 1, 1)
    return [min_v + i * step for i in range(n)]


def _ols_fit(xs: list[float], ys: list[float]):
    """Pure-Python closed-form least squares. Returns (slope, intercept) or
    None when there are fewer than 2 points or x has zero variance."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _interp_color(t: float, c0: tuple[int, int, int], c1: tuple[int, int, int]) -> str:
    t = max(0.0, min(1.0, t))
    r = round(c0[0] + (c1[0] - c0[0]) * t)
    g = round(c0[1] + (c1[1] - c0[1]) * t)
    b = round(c0[2] + (c1[2] - c0[2]) * t)
    return f"rgb({r},{g},{b})"


def bar_chart_svg(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    value_suffix: str = "",
    horizontal: bool = True,
    highlight_index: int | None = None,
    width: int = CHART_WIDTH,
    height: int | None = None,
) -> str:
    if not labels or not values or len(labels) != len(values):
        return _wrap_chart(title, no_data_svg(width, height or CHART_HEIGHT))

    n = len(values)
    max_v = max(max(values), 0.0)
    min_v = min(min(values), 0.0)
    domain = (min_v, max_v if max_v != min_v else min_v + 1.0)

    if horizontal:
        row_h = 26
        h = height or max(n * row_h + MARGIN["top"] + MARGIN["bottom"], 120)
        plot_left = MARGIN["left"] + 120
        plot_right = width - MARGIN["right"]
        plot_w = max(plot_right - plot_left, 10)
        scale = _make_scale(domain, (0, plot_w))
        zero_x = plot_left + scale(0.0)

        parts = [_svg_open(width, h, title)]
        parts.append(
            f'<line x1="{zero_x:.1f}" y1="{MARGIN["top"] - 4}" x2="{zero_x:.1f}" '
            f'y2="{h - MARGIN["bottom"] + 10:.1f}" stroke="{COLOR_GRID}"/>'
        )
        for i, (label, value) in enumerate(zip(labels, values)):
            y = MARGIN["top"] + i * row_h
            x0 = plot_left + scale(min(0.0, value))
            x1 = plot_left + scale(max(0.0, value))
            bar_w = max(x1 - x0, 0.5)
            fill = COLOR_SECONDARY if highlight_index is not None and i == highlight_index else COLOR_PRIMARY
            mid_y = y + (row_h - 6) / 2 + 4
            parts.append(
                f'<rect x="{x0:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{row_h - 6}" fill="{fill}" rx="2"/>'
                f'<text x="{plot_left - 8:.1f}" y="{mid_y:.1f}" text-anchor="end" '
                f'font-size="11" fill="{COLOR_AXIS_TEXT}">{_esc(str(label)[:28])}</text>'
                f'<text x="{x1 + 4:.1f}" y="{mid_y:.1f}" '
                f'font-size="11" fill="{COLOR_AXIS_TEXT}">{_fmt(value)}{_esc(value_suffix)}</text>'
            )
        parts.append("</svg>")
        return _wrap_chart(title, "".join(parts))

    h = height or CHART_HEIGHT
    plot_left = MARGIN["left"]
    plot_right = width - MARGIN["right"]
    plot_top = MARGIN["top"]
    plot_bottom = h - MARGIN["bottom"]
    plot_h = max(plot_bottom - plot_top, 10)
    col_w = (plot_right - plot_left) / n
    scale = _make_scale(domain, (0, plot_h))
    zero_y = plot_bottom - scale(0.0)

    parts = [_svg_open(width, h, title)]
    parts.append(f'<line x1="{plot_left}" y1="{zero_y:.1f}" x2="{plot_right}" y2="{zero_y:.1f}" stroke="{COLOR_GRID}"/>')
    for i, (label, value) in enumerate(zip(labels, values)):
        x = plot_left + i * col_w
        y0 = plot_bottom - scale(min(0.0, value))
        y1 = plot_bottom - scale(max(0.0, value))
        bar_h = max(y0 - y1, 0.5)
        fill = COLOR_SECONDARY if highlight_index is not None and i == highlight_index else COLOR_PRIMARY
        parts.append(
            f'<rect x="{x + col_w * 0.1:.1f}" y="{y1:.1f}" width="{col_w * 0.8:.1f}" height="{bar_h:.1f}" fill="{fill}" rx="2"/>'
            f'<text x="{x + col_w / 2:.1f}" y="{plot_bottom + 14:.1f}" text-anchor="middle" '
            f'font-size="10" fill="{COLOR_AXIS_TEXT}">{_esc(str(label)[:10])}</text>'
        )
    parts.append("</svg>")
    return _wrap_chart(title, "".join(parts))


def scatter_plot_svg(
    x: list[float],
    y: list[float],
    *,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    trend_line: bool = True,
    quadrant_shading: bool = False,
    split_x: float = 0.0,
    split_y: float = 0.0,
    point_labels: list[str] | None = None,
    width: int = CHART_WIDTH,
    height: int = CHART_HEIGHT,
) -> str:
    if not x or not y or len(x) != len(y):
        return _wrap_chart(title, no_data_svg(width, height))

    pts = list(zip(x, y))
    if len(pts) > 1500:
        step = max(len(pts) // 1500, 1)
        plotted = pts[::step]
        subsample_note = f"Showing a representative subsample of {len(plotted)} of {len(pts)} features."
    else:
        plotted = pts
        subsample_note = ""

    plot_left = MARGIN["left"]
    plot_right = width - MARGIN["right"]
    plot_top = MARGIN["top"]
    plot_bottom = height - MARGIN["bottom"]

    x_min = min(min(x), split_x)
    x_max = max(max(x), split_x)
    y_min = min(min(y), split_y)
    y_max = max(max(y), split_y)
    x_pad = (x_max - x_min) * 0.08 or 1.0
    y_pad = (y_max - y_min) * 0.08 or 1.0

    sx = _make_scale((x_min - x_pad, x_max + x_pad), (plot_left, plot_right))
    sy = _make_scale((y_min - y_pad, y_max + y_pad), (plot_bottom, plot_top))

    parts = [_svg_open(width, height, title)]

    if quadrant_shading:
        sx0, sy0 = sx(split_x), sy(split_y)
        parts.append(
            f'<rect x="{plot_left:.1f}" y="{plot_top:.1f}" width="{sx0 - plot_left:.1f}" height="{sy0 - plot_top:.1f}" fill="{COLOR_QUADRANT_NEG}"/>'
            f'<rect x="{sx0:.1f}" y="{plot_top:.1f}" width="{plot_right - sx0:.1f}" height="{sy0 - plot_top:.1f}" fill="{COLOR_QUADRANT_POS}"/>'
            f'<rect x="{plot_left:.1f}" y="{sy0:.1f}" width="{sx0 - plot_left:.1f}" height="{plot_bottom - sy0:.1f}" fill="{COLOR_QUADRANT_POS}"/>'
            f'<rect x="{sx0:.1f}" y="{sy0:.1f}" width="{plot_right - sx0:.1f}" height="{plot_bottom - sy0:.1f}" fill="{COLOR_QUADRANT_NEG}"/>'
        )

    for ty in _axis_ticks(y_min - y_pad, y_max + y_pad, 5):
        ty_px = sy(ty)
        parts.append(f'<line x1="{plot_left}" y1="{ty_px:.1f}" x2="{plot_right}" y2="{ty_px:.1f}" stroke="{COLOR_GRID}" stroke-dasharray="2,3"/>')
        parts.append(f'<text x="{plot_left - 6:.1f}" y="{ty_px + 3:.1f}" text-anchor="end" font-size="9" fill="{COLOR_AXIS_TEXT}">{_fmt(ty)}</text>')

    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="{COLOR_AXIS_TEXT}"/>'
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="{COLOR_AXIS_TEXT}"/>'
    )

    labels = point_labels or []
    for i, (px, py) in enumerate(plotted):
        title_attr = f"<title>{_esc(labels[i])}</title>" if i < len(labels) else ""
        parts.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="2.6" fill="{COLOR_PRIMARY}" opacity="0.65">{title_attr}</circle>')

    if trend_line:
        fit = _ols_fit(x, y)
        if fit is not None:
            slope, intercept = fit
            x0v, x1v = x_min - x_pad, x_max + x_pad
            y0v = slope * x0v + intercept
            y1v = slope * x1v + intercept
            parts.append(
                f'<line x1="{sx(x0v):.1f}" y1="{sy(y0v):.1f}" x2="{sx(x1v):.1f}" y2="{sy(y1v):.1f}" '
                f'stroke="{COLOR_TREND}" stroke-width="2" stroke-dasharray="6,3"/>'
            )

    if x_label:
        parts.append(
            f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-size="11" fill="{COLOR_AXIS_TEXT}">{_esc(x_label)}</text>'
        )
    if y_label:
        mid_y = (plot_top + plot_bottom) / 2
        parts.append(
            f'<text x="14" y="{mid_y:.1f}" text-anchor="middle" font-size="11" fill="{COLOR_AXIS_TEXT}" '
            f'transform="rotate(-90 14 {mid_y:.1f})">{_esc(y_label)}</text>'
        )

    parts.append("</svg>")
    return _wrap_chart(title, "".join(parts), subsample_note)


def line_chart_svg(
    x: list[float],
    series: dict[str, list[float]],
    *,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    markers: list[tuple[float, float, str]] | None = None,
    width: int = CHART_WIDTH,
    height: int = CHART_HEIGHT,
) -> str:
    if not x or not series or not any(series.values()):
        return _wrap_chart(title, no_data_svg(width, height))

    plot_left = MARGIN["left"]
    plot_right = width - MARGIN["right"]
    plot_top = MARGIN["top"] + (16 if len(series) > 1 else 0)
    plot_bottom = height - MARGIN["bottom"]

    all_y = [v for vals in series.values() for v in vals]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.08 or 1.0
    x_domain = (x_min, x_max) if x_min != x_max else (x_min - 1, x_max + 1)

    sx = _make_scale(x_domain, (plot_left, plot_right))
    sy = _make_scale((y_min - y_pad, y_max + y_pad), (plot_bottom, plot_top))

    parts = [_svg_open(width, height, title)]

    for ty in _axis_ticks(y_min - y_pad, y_max + y_pad, 5):
        ty_px = sy(ty)
        parts.append(f'<line x1="{plot_left}" y1="{ty_px:.1f}" x2="{plot_right}" y2="{ty_px:.1f}" stroke="{COLOR_GRID}" stroke-dasharray="2,3"/>')
        parts.append(f'<text x="{plot_left - 6:.1f}" y="{ty_px + 3:.1f}" text-anchor="end" font-size="9" fill="{COLOR_AXIS_TEXT}">{_fmt(ty)}</text>')

    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="{COLOR_AXIS_TEXT}"/>'
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="{COLOR_AXIS_TEXT}"/>'
    )

    legend_parts = []
    for i, (name, vals) in enumerate(series.items()):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        n_pts = min(len(x), len(vals))
        pts = " ".join(f"{sx(x[k]):.1f},{sy(vals[k]):.1f}" for k in range(n_pts))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        if name:
            lx = plot_left + i * 140
            legend_parts.append(
                f'<circle cx="{lx}" cy="{MARGIN["top"] - 6}" r="4" fill="{color}"/>'
                f'<text x="{lx + 8}" y="{MARGIN["top"] - 2}" font-size="10" fill="{COLOR_AXIS_TEXT}">{_esc(name)}</text>'
            )
    if len(series) > 1:
        parts.extend(legend_parts)

    for mx, my, mlabel in markers or []:
        parts.append(
            f'<circle cx="{sx(mx):.1f}" cy="{sy(my):.1f}" r="4" fill="{COLOR_TREND}"/>'
            f'<text x="{sx(mx) + 6:.1f}" y="{sy(my) - 6:.1f}" font-size="10" fill="{COLOR_TREND}">{_esc(mlabel)}</text>'
        )

    if x_label:
        parts.append(
            f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 8}" text-anchor="middle" '
            f'font-size="11" fill="{COLOR_AXIS_TEXT}">{_esc(x_label)}</text>'
        )
    if y_label:
        mid_y = (plot_top + plot_bottom) / 2
        parts.append(
            f'<text x="14" y="{mid_y:.1f}" text-anchor="middle" font-size="11" fill="{COLOR_AXIS_TEXT}" '
            f'transform="rotate(-90 14 {mid_y:.1f})">{_esc(y_label)}</text>'
        )

    parts.append("</svg>")
    return _wrap_chart(title, "".join(parts))


def histogram_svg(
    values: list[float],
    *,
    bins: int = 20,
    observed: float | None = None,
    x_label: str = "",
    title: str = "",
    width: int = CHART_WIDTH,
    height: int = CHART_HEIGHT,
) -> str:
    if not values:
        return _wrap_chart(title, no_data_svg(width, height))

    v_min, v_max = min(values), max(values)
    if v_max == v_min:
        v_max = v_min + 1.0
    bin_width = (v_max - v_min) / bins
    counts = [0] * bins
    for v in values:
        idx = min(max(int((v - v_min) / bin_width), 0), bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1

    plot_left = MARGIN["left"]
    plot_right = width - MARGIN["right"]
    plot_top = MARGIN["top"]
    plot_bottom = height - MARGIN["bottom"]
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    bar_w = plot_w / bins

    parts = [_svg_open(width, height, title)]
    for i, count in enumerate(counts):
        bar_h = (count / max_count) * plot_h if max_count else 0.0
        x = plot_left + i * bar_w
        y = plot_bottom - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 1, 0.5):.1f}" '
            f'height="{bar_h:.1f}" fill="{COLOR_PRIMARY}" opacity="0.85"/>'
        )
    parts.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="{COLOR_AXIS_TEXT}"/>')

    if observed is not None:
        obs_clamped = max(v_min, min(observed, v_max))
        obs_x = plot_left + ((obs_clamped - v_min) / (v_max - v_min)) * plot_w
        parts.append(
            f'<line x1="{obs_x:.1f}" y1="{plot_top}" x2="{obs_x:.1f}" y2="{plot_bottom}" '
            f'stroke="{COLOR_TREND}" stroke-width="2.5" stroke-dasharray="6,3"/>'
            f'<text x="{obs_x + 4:.1f}" y="{plot_top + 12}" fill="{COLOR_TREND}" font-size="11" font-weight="bold">Observed</text>'
        )

    parts.append(f'<text x="{plot_left}" y="{height - 8}" font-size="10" fill="{COLOR_AXIS_TEXT}">{_fmt(v_min)}</text>')
    parts.append(f'<text x="{plot_right - 40:.1f}" y="{height - 8}" font-size="10" fill="{COLOR_AXIS_TEXT}">{_fmt(v_max)}</text>')
    if x_label:
        parts.append(
            f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{plot_top - 6}" text-anchor="middle" '
            f'font-size="11" fill="{COLOR_AXIS_TEXT}">{_esc(x_label)}</text>'
        )

    parts.append("</svg>")
    return _wrap_chart(title, "".join(parts))


def heatmap_table_svg(
    row_labels: list[str],
    col_labels: list[str],
    matrix: list[list[float]],
    *,
    title: str = "",
    cell_format: str = "{:.0f}",
    width: int | None = None,
    height: int | None = None,
) -> str:
    if not matrix or not row_labels or not col_labels:
        return _wrap_chart(title, no_data_svg(width or CHART_WIDTH, height or CHART_HEIGHT))

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    label_w = 110
    cell_w = 70
    cell_h = 34
    w = width or (label_w + n_cols * cell_w + MARGIN["right"])
    h = height or (30 + n_rows * cell_h + MARGIN["bottom"])

    flat = [v for row in matrix for v in row]
    max_v = (max(flat) if flat else 0.0) or 1.0

    parts = [_svg_open(w, h, title)]
    for j, col in enumerate(col_labels):
        x = label_w + j * cell_w + cell_w / 2
        parts.append(f'<text x="{x:.1f}" y="18" text-anchor="middle" font-size="10" fill="{COLOR_AXIS_TEXT}">{_esc(str(col)[:10])}</text>')

    for i, row_label in enumerate(row_labels):
        y = 30 + i * cell_h
        parts.append(
            f'<text x="{label_w - 8:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="{COLOR_AXIS_TEXT}">{_esc(str(row_label)[:14])}</text>'
        )
        row = matrix[i] if i < len(matrix) else []
        for j in range(n_cols):
            value = row[j] if j < len(row) else 0
            t = value / max_v
            color = _interp_color(t, HEATMAP_LOW, HEATMAP_HIGH)
            text_color = "#ffffff" if t > 0.55 else COLOR_AXIS_TEXT
            x = label_w + j * cell_w
            try:
                cell_text = cell_format.format(value)
            except (ValueError, TypeError):
                cell_text = str(value)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{color}"/>'
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="11" fill="{text_color}">{_esc(cell_text)}</text>'
            )
    parts.append("</svg>")
    return _wrap_chart(title, "".join(parts))


def rose_diagram_svg(
    angle_degrees: float,
    *,
    magnitude: float = 1.0,
    title: str = "",
    width: int = 320,
    height: int = 320,
) -> str:
    cx, cy = width / 2, height / 2
    radius = min(width, height) / 2 - 30

    parts = [_svg_open(width, height, title)]
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius:.1f}" fill="none" stroke="{COLOR_GRID}"/>')
    for frac in (0.33, 0.66, 1.0):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius * frac:.1f}" fill="none" stroke="{COLOR_GRID}" stroke-dasharray="2,3"/>')
    for deg, label in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        rad = math.radians(deg - 90)
        lx = cx + (radius + 14) * math.cos(rad)
        ly = cy + (radius + 14) * math.sin(rad)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="11" fill="{COLOR_AXIS_TEXT}">{label}</text>')

    rad = math.radians(angle_degrees - 90)
    r = radius * max(0.0, min(magnitude, 1.0))
    x1, y1 = cx + r * math.cos(rad), cy + r * math.sin(rad)
    x2, y2 = cx - r * math.cos(rad), cy - r * math.sin(rad)
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{COLOR_TREND}" stroke-width="3"/>')
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{COLOR_AXIS_TEXT}"/>')
    parts.append("</svg>")
    return _wrap_chart(title, "".join(parts))


def lorenz_curve_svg(
    cum_pop_share: list[float],
    cum_value_share: list[float],
    *,
    gini: float | None = None,
    title: str = "",
    width: int = CHART_WIDTH,
    height: int = CHART_HEIGHT,
) -> str:
    if not cum_pop_share or not cum_value_share or len(cum_pop_share) != len(cum_value_share):
        return _wrap_chart(title, no_data_svg(width, height))

    plot_left = MARGIN["left"]
    plot_right = width - MARGIN["right"]
    plot_top = MARGIN["top"]
    plot_bottom = height - MARGIN["bottom"]

    sx = _make_scale((0.0, 1.0), (plot_left, plot_right))
    sy = _make_scale((0.0, 1.0), (plot_bottom, plot_top))
    pts = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in zip(cum_pop_share, cum_value_share))

    parts = [_svg_open(width, height, title)]
    parts.append(
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="{COLOR_AXIS_TEXT}"/>'
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="{COLOR_AXIS_TEXT}"/>'
        f'<line x1="{sx(0.0):.1f}" y1="{sy(0.0):.1f}" x2="{sx(1.0):.1f}" y2="{sy(1.0):.1f}" stroke="{COLOR_GRID}" stroke-dasharray="4,3"/>'
        f'<polyline points="{pts}" fill="none" stroke="{COLOR_PRIMARY}" stroke-width="2.5"/>'
    )
    if gini is not None:
        parts.append(f'<text x="{plot_left + 6}" y="{plot_top + 14}" font-size="11" fill="{COLOR_AXIS_TEXT}">Gini = {_fmt(gini)}</text>')
    parts.append(
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        f'font-size="11" fill="{COLOR_AXIS_TEXT}">Cumulative population share</text>'
    )
    parts.append("</svg>")
    return _wrap_chart(title, "".join(parts))


def kpi_card_row_html(cards: list[dict]) -> str:
    """cards: [{"label", "value", "sublabel"="", "tone"="neutral"|"good"|"warn"|"bad"}]"""
    tiles = []
    for card in cards:
        tone = card.get("tone", "neutral")
        tone_class = f" tone-{tone}" if tone in ("good", "warn", "bad") else ""
        sublabel = card.get("sublabel", "")
        sub_html = f'<p class="kpi-sublabel">{_esc(sublabel)}</p>' if sublabel else ""
        tiles.append(
            f'<div class="kpi-card{tone_class}">'
            f'<p class="kpi-label">{_esc(card.get("label", ""))}</p>'
            f'<p class="kpi-value">{_esc(card.get("value", ""))}</p>'
            f"{sub_html}"
            f"</div>"
        )
    return f'<div class="kpi-row">{"".join(tiles)}</div>'
