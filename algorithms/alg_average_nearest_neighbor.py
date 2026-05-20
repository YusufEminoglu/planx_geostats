# -*- coding: utf-8 -*-
"""Average Nearest Neighbor (ANN) Processing Algorithm."""
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
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml
)

from ..core.stats_engines import calculate_average_nearest_neighbor

logger = logging.getLogger("PlanX GeoStats Lab")


class AverageNearestNeighborAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    STUDY_AREA = "STUDY_AREA"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "average_nearest_neighbor"

    def displayName(self) -> str:
        return "Average Nearest Neighbor"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def createInstance(self):
        return AverageNearestNeighborAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Measures the distance from each feature centroid to its nearest neighbor's centroid. "
            "It then calculates the average of all these nearest neighbor distances.\n\n"
            "If the average distance is less than the average for a hypothetical random distribution, "
            "the distribution of the features being analyzed is considered clustered. If the average "
            "distance is greater, the features are considered dispersed.\n\n"
            "Outputs an HTML report detailing the Nearest Neighbor Ratio, observed mean distance, "
            "expected mean distance, z-score, and p-value."
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
                self.STUDY_AREA,
                "Study area (optional, in map units squared)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0,
                optional=True
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
                "Average Nearest Neighbor diagnostic report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        study_area_param = self.parameterAsDouble(parameters, self.STUDY_AREA, context)
        study_area = study_area_param if study_area_param > 0 else None

        # Resolve output HTML file path
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "ann_report.html")

        # Extract centroids
        x_coords = []
        y_coords = []

        feedback.pushInfo("Extracting feature coordinates...")
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

        if len(x_coords) <= 1:
            raise QgsProcessingException("At least 2 valid features with geometries are required for Average Nearest Neighbor analysis.")

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)

        feedback.pushInfo("Calculating nearest neighbor distances...")
        obs_mean, exp_mean, nn_ratio, z, p, calculated_area = calculate_average_nearest_neighbor(
            x_arr, y_arr, study_area
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, len(x_coords), obs_mean, exp_mean, nn_ratio, z, p, calculated_area
        )

        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path
        }

    def write_html_report(self, path: str, n: int, obs: float, exp: float, ratio: float, z: float, p: float, area: float):
        # Interpretation
        is_significant = p < 0.05
        if is_significant:
            if ratio < 1.0:
                pattern = "Clustered"
                color = "#e31a1c"
                desc = (
                    "Given the z-score of {:.2f}, there is a less than 5% likelihood that this "
                    "clustered pattern could be the result of random chance."
                ).format(z)
            else:
                pattern = "Dispersed"
                color = "#1f78b4"
                desc = (
                    "Given the z-score of {:.2f}, there is a less than 5% likelihood that this "
                    "dispersed pattern could be the result of random chance."
                ).format(z)
        else:
            pattern = "Random"
            color = "#718096"
            desc = (
                "Given the z-score of {:.2f}, the spatial pattern of features "
                "appears to be the result of random chance (no significant clustering or dispersion)."
            ).format(z)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Average Nearest Neighbor Report</title>
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
    .interpretation-box {{
        background-color: #f8fafc;
        border-left: 5px solid {color};
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 30px;
    }}
    .status-title {{
        font-size: 1.3rem;
        font-weight: 800;
        color: {color};
        margin: 0 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .status-desc {{
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
        <h1>Average Nearest Neighbor Summary</h1>
        <p class="subtitle">Sample size: <strong>{n}</strong> | Study Area: <strong>{area:.2f}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">Spatial Pattern: {pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <table>
        <thead>
            <tr>
                <th>Nearest Neighbor Metric</th>
                <th>Statistical Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="metric-name">Observed Mean Distance</td>
                <td class="metric-val">{obs:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Expected Mean Distance</td>
                <td class="metric-val">{exp:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Nearest Neighbor Ratio</td>
                <td class="metric-val">{ratio:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">z-score</td>
                <td class="metric-val">{z:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">p-value</td>
                <td class="metric-val">{p:.6f}</td>
            </tr>
        </tbody>
    </table>

    <footer>
        Generated by PlanX GeoStats Lab spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
