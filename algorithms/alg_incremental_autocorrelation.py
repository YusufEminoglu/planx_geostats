# -*- coding: utf-8 -*-
"""Incremental Spatial Autocorrelation Processing Algorithm."""
from __future__ import annotations

import logging
import os
import tempfile
import html
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml
)

from ..core.stats_engines import calculate_incremental_autocorrelation
from ..core.analysis_diagnostics import crs_unit_warning, numeric_quality_summary, push_diagnostics

logger = logging.getLogger("PlanX GeoStats Lab")


class IncrementalAutocorrelationAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    START_DISTANCE = "START_DISTANCE"
    DISTANCE_INCREMENT = "DISTANCE_INCREMENT"
    N_INCREMENTS = "N_INCREMENTS"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "incremental_spatial_autocorrelation"

    def displayName(self) -> str:
        return "Incremental Spatial Autocorrelation"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def createInstance(self):
        return IncrementalAutocorrelationAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Measures spatial autocorrelation (Global Moran's I) at multiple "
            "distance increments to identify the distance where clustering "
            "is most pronounced (peak z-score).\n\n"
            "This is essential for selecting an appropriate distance band "
            "for other spatial statistics tools like Hot Spot Analysis."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD,
                "Input field (numeric only)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.START_DISTANCE,
                "Starting distance (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=500.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_INCREMENT,
                "Distance increment (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=500.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_INCREMENTS,
                "Number of distance increments",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=3,
                maxValue=50
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output HTML report",
                fileFilter="HTML files (*.html)",
                optional=True
            )
        )
        self.addOutput(
            QgsProcessingOutputHtml(
                "HTML_REPORT_OUT",
                "Incremental autocorrelation report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        start_dist = self.parameterAsDouble(parameters, self.START_DISTANCE, context)
        dist_inc = self.parameterAsDouble(parameters, self.DISTANCE_INCREMENT, context)
        n_inc = self.parameterAsInt(parameters, self.N_INCREMENTS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "incremental_autocorrelation.html")

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Field '{field_name}' not found.")

        x_coords, y_coords, values = [], [], []
        value_dict = {}
        skipped = 0

        feedback.pushInfo("Extracting features...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            geom = f.geometry()
            if geom.isEmpty():
                skipped += 1
                continue
            val = f.attribute(field_idx)
            if val is None or val == QVariant() or str(val) == 'NULL':
                skipped += 1
                continue
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                skipped += 1
                continue
            centroid = geom.centroid().asPoint()
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())
            values.append(val_f)
            value_dict[f.id()] = val_f
            feedback.setProgress(int(20 * (idx / total)))

        n_feats = len(x_coords)
        numeric_summary = numeric_quality_summary(source.featureCount(), value_dict, values)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, None, crs_warning)
        if n_feats < 4:
            raise QgsProcessingException(f"Insufficient features ({n_feats}). At least 4 required.")
        if numeric_summary["is_constant"]:
            raise QgsProcessingException("Incremental autocorrelation requires variation in the target field; all valid values are identical.")

        feedback.pushInfo(f"Computing Moran's I at {n_inc} distance increments...")
        results = calculate_incremental_autocorrelation(
            np.array(x_coords), np.array(y_coords), np.array(values),
            start_dist, dist_inc, n_inc
        )

        # Find peak z-score
        peak = max(results, key=lambda r: abs(r["z_score"]))

        feedback.pushInfo(f"Peak z-score: {peak['z_score']:.4f} at distance {peak['distance']:.2f}")
        feedback.pushInfo(
            f"Peak neighborhood support: min={peak['min_neighbors']}, "
            f"median={peak['median_neighbors']:.2f}, max={peak['max_neighbors']}, "
            f"isolated={peak['isolated_count']}."
        )
        feedback.pushInfo("Generating HTML report...")
        self._write_html(html_path, field_name, n_feats, results, peak, skipped, crs_warning)

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, field, n, results, peak, skipped, crs_warning):
        # SVG line chart
        svg_w, svg_h = 600, 250
        pad_l, pad_b, pad_t, pad_r = 60, 40, 30, 20
        plot_w = svg_w - pad_l - pad_r
        plot_h = svg_h - pad_t - pad_b

        dists = [r["distance"] for r in results]
        zs = [r["z_score"] for r in results]
        d_min, d_max = min(dists), max(dists)
        z_min, z_max = min(zs), max(zs)
        z_range = z_max - z_min if z_max != z_min else 1.0
        d_range = d_max - d_min if d_max != d_min else 1.0

        def to_svg(d, z):
            sx = pad_l + (d - d_min) / d_range * plot_w
            sy = pad_t + plot_h - ((z - z_min) / z_range * plot_h)
            return sx, sy

        # Polyline points
        pts = " ".join(f"{to_svg(d, z)[0]:.1f},{to_svg(d, z)[1]:.1f}" for d, z in zip(dists, zs))

        # Dots
        dots = ""
        for d, z in zip(dists, zs):
            sx, sy = to_svg(d, z)
            dots += f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" fill="#4299e1"/>'

        # Peak marker
        px, py = to_svg(peak["distance"], peak["z_score"])
        peak_marker = f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="none" stroke="#e31a1c" stroke-width="2.5"/>'

        # Table rows
        rows = ""
        for r in results:
            is_peak = "Peak" if r["distance"] == peak["distance"] else ""
            sig = "Significant" if r["p_value"] < 0.05 else ""
            rows += f"""<tr>
                <td class="metric-val">{r['distance']:.2f} {is_peak}</td>
                <td class="metric-val">{r['morans_i']:.6f}</td>
                <td class="metric-val">{r['z_score']:.4f}</td>
                <td class="metric-val">{r['p_value']:.4f} {sig}</td>
                <td class="metric-val">{r['min_neighbors']} / {r['median_neighbors']:.1f} / {r['max_neighbors']}</td>
                <td class="metric-val">{r['isolated_count']}</td>
            </tr>"""

        if peak["isolated_count"] > 0:
            next_action = "Increase the starting distance or increment because the peak still contains isolated observations."
        elif peak["max_neighbors"] >= n - 1:
            next_action = "Test smaller distances around the peak because the graph is close to fully connected."
        elif peak["p_value"] < 0.05:
            next_action = "Use the peak distance as a candidate threshold for Global Moran, General G, Gi*, or Local Moran tools."
        else:
            next_action = "Do not rely on a single peak; compare several distance bands and review the study-area scale."
        crs_block = f"<div class=\"note\"><strong>CRS warning:</strong> {html.escape(crs_warning)}</div>" if crs_warning else ""

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Incremental Autocorrelation</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #2d3748; background: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 760px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,.1); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: .95rem; }}
    .peak-box {{ background: #f0fff4; border-left: 5px solid #2b9348; padding: 15px 20px; border-radius: 4px; margin-bottom: 25px; }}
    .peak-box strong {{ color: #22543d; }}
    .note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
    .next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
    h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: .85rem; }}
    th {{ background: #ebf8ff; color: #2b6cb0; font-weight: 700; text-transform: uppercase; font-size: .7rem; letter-spacing: .05em; }}
    .metric-val {{ font-family: monospace; font-weight: 600; }}
    footer {{ margin-top: 35px; border-top: 1px solid #edf2f7; padding-top: 12px; font-size: .8rem; color: #a0aec0; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Incremental Spatial Autocorrelation</h1>
        <p class="subtitle">Attribute: <strong>{html.escape(field)}</strong> | Features: <strong>{n}</strong> | Increments: <strong>{len(results)}</strong> | Skipped: <strong>{skipped}</strong></p>
    </header>

    <div class="peak-box">
        <strong>Peak Clustering Distance: {peak['distance']:.2f} map units</strong> (z-score = {peak['z_score']:.4f}, p = {peak['p_value']:.4f})
        <br>Neighborhood support at peak: min / median / max neighbors = <strong>{peak['min_neighbors']} / {peak['median_neighbors']:.1f} / {peak['max_neighbors']}</strong>; isolated observations = <strong>{peak['isolated_count']}</strong>.
    </div>

    <h2>Executive Summary</h2>
    <p>Incremental Spatial Autocorrelation scans multiple distance bands to identify the scale where global clustering is strongest. Treat the peak as a candidate analysis distance, not as an automatic final setting; compare it with domain knowledge and neighborhood support.</p>
    {crs_block}

    <svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
        <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#e2e8f0"/>
        <polyline points="{pts}" fill="none" stroke="#4299e1" stroke-width="2.5" stroke-linejoin="round"/>
        {dots}
        {peak_marker}
        <text x="{pad_l - 10}" y="{pad_t - 8}" fill="#718096" font-size="11" text-anchor="end">z-score</text>
        <text x="{pad_l}" y="{svg_h - 5}" fill="#718096" font-size="10">{d_min:.0f}</text>
        <text x="{svg_w - pad_r}" y="{svg_h - 5}" fill="#718096" font-size="10" text-anchor="end">{d_max:.0f}</text>
        <text x="{pad_l - 5}" y="{pad_t + 10}" fill="#718096" font-size="10" text-anchor="end">{z_max:.2f}</text>
        <text x="{pad_l - 5}" y="{pad_t + plot_h}" fill="#718096" font-size="10" text-anchor="end">{z_min:.2f}</text>
    </svg>

    <table>
        <thead><tr><th>Distance</th><th>Moran's I</th><th>z-score</th><th>p-value</th><th>Min / Median / Max Neighbors</th><th>Isolated</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>

    <h2>Recommended Next Action</h2>
    <div class="next-action">{html.escape(next_action)}</div>

    <h2>Assumptions and Caveats</h2>
    <ul>
        <li>Distance increments are evaluated using centroid distances and map units.</li>
        <li>A high z-score can be unreliable when the neighborhood graph contains many isolated observations.</li>
        <li>A nearly fully connected graph can hide local structure and make several distances look similar.</li>
    </ul>

    <footer>Generated by PlanX GeoStats Lab spatial statistics engine. Peak marks the strongest absolute z-score; Significant means p &lt; 0.05.</footer>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
