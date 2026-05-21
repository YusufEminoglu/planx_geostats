# -*- coding: utf-8 -*-
"""Getis-Ord General G Processing Algorithm."""
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

from ..core.stats_engines import calculate_general_g
from ..core.analysis_diagnostics import (
    caveats_html,
    crs_unit_warning,
    diagnostics_html,
    neighbor_summary,
    numeric_quality_summary,
    push_diagnostics,
)

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class GeneralGAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    DISTANCE_BAND = "DISTANCE_BAND"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "general_g_autocorrelation"

    def displayName(self) -> str:
        return "High/Low Clustering (Getis-Ord General G)"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def icon(self):
        return algorithm_icon("general_g_autocorrelation")

    def createInstance(self):
        return GeneralGAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Measures the degree of clustering for either high values or low values over "
            "the study area using the Getis-Ord General G statistic.\n\n"
            "Calculates the observed General G index, compares it with the expected General G index, "
            "and derives a z-score and p-value representing statistical significance under the "
            "randomization assumption.\n\n"
            "Outputs a diagnostic HTML report with detailed metrics and spatial pattern interpretation."
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
                "High/Low Clustering (General G) report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)

        # Resolve output HTML file path
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            temp_dir = tempfile.gettempdir()
            html_path = os.path.join(temp_dir, "general_g_report.html")

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Field '{field_name}' not found.")

        # Extract features and coordinates
        centroids = {}
        id_order = []
        values = {}

        feedback.pushInfo("Extracting attributes and geometries...")
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
                if val_f < 0:
                    feedback.pushWarning(
                        "General G calculations require non-negative values. "
                        "A negative value was encountered."
                    )
            except (ValueError, TypeError):
                continue

            fid = f.id()
            centroids[fid] = geom.centroid().asPoint()
            id_order.append(fid)
            values[fid] = val_f

            feedback.setProgress(int(20 * (idx / total)))

        n_feats = len(id_order)
        val_array = np.array([values[fid] for fid in id_order])
        numeric_summary = numeric_quality_summary(source.featureCount(), values, val_array)
        if n_feats < 4:
            raise QgsProcessingException(
                f"Insufficient features ({n_feats}) with valid numeric values. "
                "At least 4 are required for General G analysis."
            )
        if numeric_summary["is_constant"]:
            raise QgsProcessingException("General G requires variation in the target field; all valid values are identical.")

        # Construct spatial weights matrix (Binary weights within distance band)
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
                    weights[fid_i].append(1.0)  # Binary spatial weights

            feedback.setProgress(int(20 + 30 * (i / n_feats)))

        # Verify weights sum > 0
        total_w = sum(sum(w_list) for w_list in weights.values())
        neighborhood_summary = neighbor_summary(neighbors, id_order)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, neighborhood_summary, crs_warning)
        if total_w == 0:
            raise QgsProcessingException(
                "No neighbors found within the specified distance band. "
                "Please increase the distance band value."
            )

        feedback.pushInfo("Calculating Getis-Ord General G statistics...")
        obs_g, exp_g, var_g, z, p = calculate_general_g(
            values, neighbors, weights, id_order
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path,
            field_name,
            n_feats,
            distance_band,
            obs_g,
            exp_g,
            var_g,
            z,
            p,
            numeric_summary,
            neighborhood_summary,
            crs_warning,
        )

        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path
        }

    def write_html_report(
        self,
        path: str,
        field: str,
        n: int,
        db: float,
        obs: float,
        exp: float,
        var: float,
        z: float,
        p: float,
        numeric_summary: dict,
        neighborhood_summary: dict,
        crs_warning: str,
    ):
        is_significant = p < 0.05
        if is_significant:
            if z > 0:
                pattern = "Clustering of High Values"
                color = "#e31a1c"
                desc = (
                    "Given the positive z-score of {:.2f}, there is a less than 5% likelihood "
                    "that this high-value clustering (Hot Spot concentration) could be the result of random chance."
                ).format(z)
            else:
                pattern = "Clustering of Low Values"
                color = "#1f78b4"
                desc = (
                    "Given the negative z-score of {:.2f}, there is a less than 5% likelihood "
                    "that this low-value clustering (Cold Spot concentration) could be the result of random chance."
                ).format(z)
        else:
            pattern = "Random (No Significant Clustering)"
            color = "#718096"
            desc = (
                "Given the z-score of {:.2f}, the spatial distribution of values "
                "appears to be completely random (Complete Spatial Randomness)."
            ).format(z)

        if neighborhood_summary["isolated"] > 0:
            next_action = "Increase the distance band before interpreting high/low clustering; isolated observations weaken the global statistic."
        elif neighborhood_summary["all_connected"]:
            next_action = "Try a smaller distance band because the current threshold connects every feature to every other feature."
        elif is_significant and z > 0:
            next_action = "Use Gi* Hot Spot Analysis to map the specific high-value concentrations suggested by this global result."
        elif is_significant:
            next_action = "Use Gi* or Local Moran's I to locate the low-value concentrations suggested by this global result."
        else:
            next_action = "Review the threshold distance and attribute distribution before treating the absence of significance as absence of spatial structure."

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab General G Report</title>
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
    section {{
        margin: 28px 0;
    }}
    h2 {{
        color: #1a202c;
        font-size: 1.15rem;
        margin: 0 0 12px 0;
    }}
    .next-action {{
        background: #f0fff4;
        border-left: 5px solid #2f855a;
        padding: 16px 18px;
        border-radius: 4px;
    }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>High/Low Clustering (General G) Summary</h1>
        <p class="subtitle">Attribute Analyzed: <strong>{html.escape(field)}</strong> | Count: <strong>{n}</strong> | Distance Band: <strong>{db} map units</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">{pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>Getis-Ord General G evaluates whether high values or low values are globally concentrated within the selected distance band. This result is scale-sensitive: a threshold that is too small can create isolated observations, while a threshold that is too large can flatten local structure into a broad global average.</p>
    </section>

    <table>
        <thead>
            <tr>
                <th>General G Statistic</th>
                <th>Calculated Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td class="metric-name">Observed General G Index</td>
                <td class="metric-val">{obs:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">Expected General G Index</td>
                <td class="metric-val">{exp:.6f}</td>
            </tr>
            <tr>
                <td class="metric-name">General G Variance</td>
                <td class="metric-val">{var:.8f}</td>
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

    {diagnostics_html(numeric_summary, neighborhood_summary, crs_warning)}

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>

    {caveats_html("Getis-Ord General G", neighborhood_summary, numeric_summary)}

    <footer>
        Generated by PlanX GeoStats Lab global spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
