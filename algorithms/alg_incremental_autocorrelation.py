# -*- coding: utf-8 -*-
"""Incremental Spatial Autocorrelation Processing Algorithm."""
from __future__ import annotations

import logging
import os
import tempfile
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

        feedback.pushInfo("Extracting features...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            geom = f.geometry()
            if geom.isEmpty():
                continue
            val = f.attribute(field_idx)
            if val is None or val == QVariant() or str(val) == 'NULL':
                continue
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                continue
            centroid = geom.centroid().asPoint()
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())
            values.append(val_f)
            feedback.setProgress(int(20 * (idx / total)))

        n_feats = len(x_coords)
        if n_feats < 4:
            raise QgsProcessingException(f"Insufficient features ({n_feats}). At least 4 required.")

        feedback.pushInfo(f"Computing Moran's I at {n_inc} distance increments...")
        results = calculate_incremental_autocorrelation(
            np.array(x_coords), np.array(y_coords), np.array(values),
            start_dist, dist_inc, n_inc
        )

        # Find peak z-score
        peak = max(results, key=lambda r: abs(r["z_score"]))

        feedback.pushInfo(f"Peak z-score: {peak['z_score']:.4f} at distance {peak['distance']:.2f}")
        feedback.pushInfo("Generating HTML report...")
        self._write_html(html_path, field_name, n_feats, results, peak)

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, field, n, results, peak):
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
            </tr>"""

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
        <p class="subtitle">Attribute: <strong>{field}</strong> | Features: <strong>{n}</strong> | Increments: <strong>{len(results)}</strong></p>
    </header>

    <div class="peak-box">
        <strong>Peak Clustering Distance: {peak['distance']:.2f} map units</strong> (z-score = {peak['z_score']:.4f}, p = {peak['p_value']:.4f})
    </div>

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
        <thead><tr><th>Distance</th><th>Moran's I</th><th>z-score</th><th>p-value</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>

    <footer>Generated by PlanX GeoStats Lab spatial statistics engine. Peak marks the strongest absolute z-score; Significant means p &lt; 0.05.</footer>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
