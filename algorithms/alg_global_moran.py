# -*- coding: utf-8 -*-
"""Global Moran's I (Spatial Autocorrelation) Processing Algorithm."""
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
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml
)

from ..core.weights import build_weights_matrix
from ..core.stats_engines import calculate_global_moran

logger = logging.getLogger("PlanX-GeoStats")


class GlobalMoranAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "global_moran_autocorrelation"

    def displayName(self) -> str:
        return "Spatial Autocorrelation (Global Moran's I)"

    def group(self) -> str:
        return "Analyzing Patterns"

    def groupId(self) -> str:
        return "analyzing_patterns"

    def createInstance(self):
        return GlobalMoranAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Measures spatial autocorrelation based on both feature locations and feature values "
            "using the Global Moran's I statistic.\n\n"
            "Evaluates whether the pattern expressed is clustered, dispersed, or random. "
            "Generates an HTML diagnostic report showing the Moran's I index, expected index, "
            "variance, z-score, and p-value."
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
                "Target numeric field to analyze",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001
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
                "Spatial autocorrelation diagnostic report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)

        # Resolve output HTML file path
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "global_moran_report.html")

        # Validate target field
        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        field = source.fields().at(field_idx)
        if not field.isNumeric():
            raise QgsProcessingException(f"Target field '{field_name}' must be numeric.")

        feedback.pushInfo("Generating spatial weights matrix...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source,
            weight_type,
            k_neighbors=k_neighbors,
            distance_band=distance_band,
            feedback=feedback
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Extracting target field values...")
        y_dict = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val = f.attribute(field_name)
            if val is None or val == QVariant() or str(val) == 'NULL':
                continue
            try:
                y_dict[f.id()] = float(val)
            except (ValueError, TypeError):
                continue

        # Filter id_order and construct y array
        valid_id_order = [fid for fid in id_order if fid in y_dict]
        y = np.array([y_dict[fid] for fid in valid_id_order])

        if len(y) <= 3:
            raise QgsProcessingException("At least 4 valid features with numeric values are required for Global Moran's I analysis.")

        feedback.pushInfo("Calculating Global Moran's I statistics...")
        moran_i, expected_i, variance, z_score, p_value = calculate_global_moran(
            y,
            neighbors,
            weights,
            valid_id_order
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, len(y), moran_i, expected_i, variance, z_score, p_value
        )

        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path
        }

    def write_html_report(self, path: str, field_name: str, n: int, mi: float, ei: float, var: float, z: float, p: float):
        # Interpretation logic
        sig_threshold = 0.05
        is_significant = p < sig_threshold

        if is_significant:
            if z > 0:
                pattern = "Clustered"
                desc = (
                    "Given the z-score of {:.2f}, there is a less than 5% likelihood that this "
                    "clustered pattern could be the result of random chance."
                ).format(z)
                status_class = "clustered"
                status_color = "#e31a1c"
            else:
                pattern = "Dispersed"
                desc = (
                    "Given the z-score of {:.2f}, there is a less than 5% likelihood that this "
                    "dispersed pattern could be the result of random chance."
                ).format(z)
                status_class = "dispersed"
                status_color = "#1f78b4"
        else:
            pattern = "Random"
            desc = (
                "Given the z-score of {:.2f}, the spatial pattern of features "
                "appears to be the result of random chance (no significant autocorrelation)."
            ).format(z)
            status_class = "random"
            status_color = "#718096"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX-GeoStats Spatial Autocorrelation Report</title>
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
        border-left: 5px solid {status_color};
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 30px;
    }}
    .status-title {{
        font-size: 1.3rem;
        font-weight: 800;
        color: {status_color};
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
        <h1>Spatial Autocorrelation (Global Moran's I)</h1>
        <p class="subtitle">Field Analyzed: <strong>{field_name}</strong> | Feature Count: <strong>{n}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">Spatial Pattern: {pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <table>
        <thead>
            <tr>
                <th>Global Moran's I Diagnostic</th>
                <th>Statistical Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="metric-name">Moran's Index</td>
                <td class="metric-val">{mi:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Expected Index</td>
                <td class="metric-val">{ei:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Variance</td>
                <td class="metric-val">{var:.6f}</td>
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
        Generated by PlanX-GeoStats spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
