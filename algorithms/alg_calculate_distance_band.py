# -*- coding: utf-8 -*-
"""Calculate Distance Band from Neighbor Count Processing Algorithm."""
from __future__ import annotations

import logging
import os
import tempfile
import numpy as np

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml
)

from ..core.stats_engines import calculate_distance_band_stats

logger = logging.getLogger("PlanX-GeoStats")


class CalculateDistanceBandAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    NEIGHBOR_COUNT = "NEIGHBOR_COUNT"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "calculate_distance_band"

    def displayName(self) -> str:
        return "Calculate Distance Band from Neighbor Count"

    def group(self) -> str:
        return "Spatial Component Utilities"

    def groupId(self) -> str:
        return "spatial_component_utilities"

    def createInstance(self):
        return CalculateDistanceBandAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Calculates distance statistics to the N-th nearest neighbor (minimum, average, maximum, "
            "and percentiles) for all features.\n\n"
            "This tool is extremely useful for choosing a threshold distance band when running spatial "
            "autocorrelation or hot spot analysis. For example, using the maximum nearest neighbor distance "
            "as the threshold ensures that every feature has at least one neighbor in the analysis."
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
            QgsProcessingParameterNumber(
                self.NEIGHBOR_COUNT,
                "Neighbor count (N)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1
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
                "Distance band statistics report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        k_neighbors = self.parameterAsInt(parameters, self.NEIGHBOR_COUNT, context)

        # Resolve output HTML file path
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "distance_band_report.html")

        # Extract centroids
        x_coords = []
        y_coords = []

        feedback.pushInfo("Extracting feature centroids...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom.isEmpty():
                continue

            centroid = geom.centroid().asPoint()
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())
            feedback.setProgress(int(40 * (idx / total)))

        n_feats = len(x_coords)
        if n_feats <= k_neighbors:
            raise QgsProcessingException(
                f"Feature count ({n_feats}) must be greater than neighbor count ({k_neighbors}) to compute stats."
            )

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)

        feedback.pushInfo("Computing pairwise distance bands...")
        stats = calculate_distance_band_stats(x_arr, y_arr, k_neighbors)

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(html_path, n_feats, k_neighbors, stats)

        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path
        }

    def write_html_report(self, path: str, n: int, k: int, stats: dict):
        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX-GeoStats Distance Band Report</title>
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
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
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
    .advice-box {{
        background-color: #ebf8ff;
        border-left: 5px solid #3182ce;
        padding: 15px 20px;
        border-radius: 4px;
        margin-bottom: 30px;
        font-size: 0.9rem;
        color: #2b6cb0;
    }}
    .advice-box strong {{
        color: #2c5282;
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
    .metric-name {{
        font-weight: 600;
        color: #2d3748;
    }}
    .metric-val {{
        font-family: monospace;
        font-size: 1rem;
        font-weight: 600;
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
        <h1>Distance Band Statistics Summary</h1>
        <p class="subtitle">Feature count: <strong>{n}</strong> | Neighbor count: <strong>{k}</strong></p>
    </header>

    <div class="advice-box">
        <strong>💡 Recommendation:</strong> To ensure that every feature in the dataset has at least <strong>{k}</strong> neighbor(s) during analysis, select a threshold distance band of at least <strong>{stats["max"]:.6f}</strong> map units.
    </div>

    <table>
        <thead>
            <tr>
                <th>Neighbor Distance Parameter</th>
                <th>Distance (Map Units)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="metric-name">Minimum Distance to {k}-th Neighbor</td>
                <td class="metric-val">{stats["min"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Average Distance to {k}-th Neighbor</td>
                <td class="metric-val">{stats["mean"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">25th Percentile (p25)</td>
                <td class="metric-val">{stats["p25"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">50th Percentile / Median Distance</td>
                <td class="metric-val">{stats["median"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">75th Percentile (p75)</td>
                <td class="metric-val">{stats["p75"]:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Maximum Distance to {k}-th Neighbor (Ensures Connectivity)</td>
                <td class="metric-val" style="color: #c53030;">{stats["max"]:.6f}</td>
            </tr>
        </tbody>
    </table>

    <footer>
        Generated by PlanX-GeoStats spatial component utilities.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
