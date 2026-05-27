# -*- coding: utf-8 -*-
"""Attribute Randomization Sensitivity Test Processing Algorithm."""
from __future__ import annotations

import logging
import os
import tempfile
import html
import numpy as np

from qgis.core import (
    NULL,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml
)

from ..core.stats_engines import run_sensitivity_simulation
from ..core.analysis_diagnostics import (
    caveats_html,
    crs_unit_warning,
    diagnostics_html,
    neighbor_summary,
    numeric_quality_summary,
    push_diagnostics,
)
from ..core.sensitivity_audit import sensitivity_verdict
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class SensitivityTestAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    DISTANCE_BAND = "DISTANCE_BAND"
    SIMULATIONS = "SIMULATIONS"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "sensitivity_test"

    def displayName(self) -> str:
        return "Attribute Randomization Sensitivity Test"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("sensitivity_test")

    def createInstance(self):
        return SensitivityTestAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Assesses the sensitivity of Global Moran's I spatial autocorrelation to "
            "spatial randomness using Monte Carlo permutation simulations.\n\n"
            "The observed Moran's I is compared against a reference distribution generated "
            "by randomly shuffling (permuting) the attribute values across features. "
            "The result indicates whether the observed spatial pattern is statistically "
            "robust or could be an artifact of random arrangement."
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
                self.DISTANCE_BAND,
                "Distance band (threshold distance in map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SIMULATIONS,
                "Number of simulations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=999,
                minValue=99,
                maxValue=9999
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
                "Sensitivity test report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        n_sims = self.parameterAsInt(parameters, self.SIMULATIONS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "sensitivity_report.html")

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Field '{field_name}' not found.")

        # Extract features
        centroids = {}
        id_order = []
        values = {}
        skipped = 0

        feedback.pushInfo("Extracting attributes and geometries...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom is None or geom.isEmpty():
                skipped += 1
                continue

            val = f.attribute(field_idx)
            if val is None or val == NULL or str(val) == 'NULL':
                skipped += 1
                continue
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                skipped += 1
                continue

            fid = f.id()
            centroid = geometry_centroid_point(geom)
            if centroid is None:
                skipped += 1
                continue
            centroids[fid] = centroid
            id_order.append(fid)
            values[fid] = val_f
            feedback.setProgress(int(15 * (idx / total)))

        n_feats = len(id_order)
        val_array = np.array([values[fid] for fid in id_order])
        numeric_summary = numeric_quality_summary(source.featureCount(), values, val_array)
        if n_feats < 4:
            raise QgsProcessingException(
                f"Insufficient features ({n_feats}). At least 4 are required."
            )
        if numeric_summary["is_constant"]:
            raise QgsProcessingException("Sensitivity testing requires variation in the input field; all valid values are identical.")

        # Construct spatial weights
        feedback.pushInfo("Constructing spatial weights matrix...")
        neighbors = {}
        weights = {}

        for i in range(n_feats):
            if feedback.isCanceled():
                break
            fid_i = id_order[i]
            pt_i = centroids[fid_i]
            neighbors[fid_i] = []
            weights[fid_i] = []
            for j in range(n_feats):
                if i == j:
                    continue
                fid_j = id_order[j]
                pt_j = centroids[fid_j]
                dist = pt_i.distance(pt_j)
                if dist <= distance_band:
                    neighbors[fid_i].append(fid_j)
                    weights[fid_i].append(1.0)
            feedback.setProgress(int(15 + 25 * (i / n_feats)))

        total_w = sum(len(wl) for wl in weights.values())
        neighborhood_summary = neighbor_summary(neighbors, id_order)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, neighborhood_summary, crs_warning)
        if total_w == 0:
            raise QgsProcessingException(
                "No neighbors found within the specified distance band."
            )

        feedback.pushInfo(f"Running {n_sims} Monte Carlo permutations...")
        results = run_sensitivity_simulation(
            val_array, neighbors, weights, id_order, n_sims
        )

        feedback.pushInfo("Generating HTML report...")
        self._write_html_report(
            html_path,
            field_name,
            n_feats,
            distance_band,
            n_sims,
            results,
            numeric_summary,
            neighborhood_summary,
            crs_warning,
            skipped,
        )

        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path
        }

    def _write_html_report(self, path: str, field: str, n: int, db: float, n_sims: int, r: dict, numeric_summary: dict, neighborhood_summary: dict, crs_warning: str, skipped: int):
        obs_i = r["observed_i"]
        sim_mean = r["simulated_mean"]
        sim_std = r["simulated_std"]
        emp_p = r["empirical_p"]
        p5 = r["percentile_5"]
        p95 = r["percentile_95"]

        interpretation = sensitivity_verdict(r, neighborhood_summary)
        verdict = interpretation["verdict"]
        verdict_color = interpretation["color"]
        verdict_desc = interpretation["description"]
        next_action = interpretation["next_action"]
        caution_items = "".join(f"<li>{html.escape(item)}</li>" for item in interpretation["cautions"])

        # Simple ASCII histogram of simulated values
        sim_vals = r["simulated_values"]
        n_bins = 20
        hist_min = min(sim_vals)
        hist_max = max(sim_vals)
        if hist_max == hist_min:
            hist_max = hist_min + 1
        bin_width = (hist_max - hist_min) / n_bins
        bins = [0] * n_bins
        for v in sim_vals:
            b = int((v - hist_min) / bin_width)
            b = min(b, n_bins - 1)
            bins[b] += 1
        max_count = max(bins) if bins else 1

        # SVG histogram
        svg_w, svg_h = 580, 200
        bar_w = svg_w / n_bins
        svg_bars = ""
        for i, count in enumerate(bins):
            bar_h = (count / max_count) * (svg_h - 30) if max_count > 0 else 0
            x = i * bar_w
            y = svg_h - 20 - bar_h
            svg_bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{bar_h:.1f}" fill="#4299e1" opacity="0.8"/>'

        # Observed line
        obs_x = ((obs_i - hist_min) / (hist_max - hist_min)) * svg_w
        obs_x = max(0, min(obs_x, svg_w))

        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Sensitivity Test Report</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #2d3748;
        background-color: #f7fafc;
        margin: 0;
        padding: 20px;
        line-height: 1.5;
    }}
    .container {{
        max-width: 760px;
        margin: 0 auto;
        background: #ffffff;
        padding: 30px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }}
    header {{
        border-bottom: 2px solid #edf2f7;
        padding-bottom: 20px;
        margin-bottom: 25px;
    }}
    h1 {{
        color: #1a202c;
        margin: 0 0 5px 0;
        font-size: 1.6rem;
    }}
    .subtitle {{
        color: #718096;
        margin: 0;
        font-size: 0.95rem;
    }}
    .verdict-box {{
        background-color: #f8fafc;
        border-left: 5px solid {verdict_color};
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 30px;
    }}
    .verdict-title {{
        font-size: 1.2rem;
        font-weight: 800;
        color: {verdict_color};
        margin: 0 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .verdict-desc {{
        color: #4a5568;
        font-size: 0.95rem;
        margin: 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 25px;
    }}
    th, td {{
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #edf2f7;
        font-size: 0.9rem;
    }}
    th {{
        background-color: #ebf8ff;
        color: #2b6cb0;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }}
    .metric-name {{ font-weight: 600; color: #2d3748; }}
    .metric-val {{ font-family: monospace; font-size: 1rem; font-weight: 600; }}
    .hist-section {{
        margin: 25px 0;
    }}
    .hist-section h3 {{
        font-size: 1rem;
        margin-bottom: 10px;
        color: #2d3748;
    }}
    h2 {{
        color: #1a202c;
        font-size: 1.15rem;
        margin: 28px 0 12px;
    }}
    .next-action {{
        background: #f0fff4;
        border-left: 5px solid #2f855a;
        padding: 16px 18px;
        border-radius: 4px;
    }}
    footer {{
        margin-top: 40px;
        border-top: 1px solid #edf2f7;
        padding-top: 15px;
        font-size: 0.8rem;
        color: #a0aec0;
        text-align: center;
    }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Sensitivity Analysis (Monte Carlo Simulation)</h1>
        <p class="subtitle">Attribute: <strong>{html.escape(field)}</strong> | Features: <strong>{n}</strong> | Distance Band: <strong>{db}</strong> | Simulations: <strong>{n_sims}</strong> | Skipped: <strong>{skipped}</strong></p>
    </header>

    <div class="verdict-box">
        <h2 class="verdict-title">{verdict}</h2>
        <p class="verdict-desc">{verdict_desc}</p>
    </div>

    <h2>Executive Summary</h2>
    <p>This sensitivity test compares the observed Moran's I against a Monte Carlo reference distribution created by randomly permuting attribute values across the same spatial structure. A robust result means the observed pattern is unusual under random reassignment; it does not by itself prove causation.</p>

    <table>
        <thead>
            <tr><th>Metric</th><th>Value</th></tr>
        </thead>
        <tbody>
            <tr><td class="metric-name">Observed Moran's I</td><td class="metric-val">{obs_i:.6f}</td></tr>
            <tr><td class="metric-name">Simulated Mean Moran's I</td><td class="metric-val">{sim_mean:.6f}</td></tr>
            <tr><td class="metric-name">Simulated Std Dev</td><td class="metric-val">{sim_std:.6f}</td></tr>
            <tr><td class="metric-name">Empirical p-value</td><td class="metric-val">{emp_p:.4f}</td></tr>
            <tr><td class="metric-name">5th Percentile (Reference Distribution)</td><td class="metric-val">{p5:.6f}</td></tr>
            <tr><td class="metric-name">95th Percentile (Reference Distribution)</td><td class="metric-val">{p95:.6f}</td></tr>
        </tbody>
    </table>

    {diagnostics_html(numeric_summary, neighborhood_summary, crs_warning)}

    <div class="hist-section">
        <h3>Reference Distribution of Simulated Moran's I Values</h3>
        <svg width="{svg_w}" height="{svg_h + 10}" viewBox="0 0 {svg_w} {svg_h + 10}">
            {svg_bars}
            <line x1="{obs_x:.1f}" y1="0" x2="{obs_x:.1f}" y2="{svg_h - 20}" stroke="#e31a1c" stroke-width="2.5" stroke-dasharray="6,3"/>
            <text x="{obs_x + 4:.1f}" y="14" fill="#e31a1c" font-size="11" font-weight="bold">Observed I</text>
            <text x="0" y="{svg_h}" fill="#718096" font-size="10">{hist_min:.4f}</text>
            <text x="{svg_w - 60}" y="{svg_h}" fill="#718096" font-size="10">{hist_max:.4f}</text>
        </svg>
    </div>

    <h2>Recommended Next Action</h2>
    <div class="next-action">{html.escape(next_action)}</div>

    <h2>Sensitivity Cautions</h2>
    <ul>{caution_items}</ul>

    {caveats_html("Attribute Randomization Sensitivity Test", neighborhood_summary, numeric_summary)}

    <footer>
        Generated by PlanX GeoStats Lab sensitivity analysis engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
