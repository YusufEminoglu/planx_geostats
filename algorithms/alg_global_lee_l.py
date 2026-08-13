# -*- coding: utf-8 -*-
"""Global Bivariate Lee's L Processing Algorithm."""
from __future__ import annotations

import os
import tempfile
import html
import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml,
)

from ..core.weights import build_weights_matrix
from ..core.advanced_stats_engines import calculate_global_lee_l
from ..core.analysis_diagnostics import (
    caveats_html,
    crs_unit_warning,
    diagnostics_html,
    neighbor_summary,
    numeric_quality_summary,
    push_diagnostics,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html

from ._icons import algorithm_icon


class GlobalLeesLAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD_X = "FIELD_X"
    FIELD_Y = "FIELD_Y"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "global_bivariate_lee_l"

    def displayName(self) -> str:
        return "Global Bivariate Lee's L"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def icon(self):
        return algorithm_icon("global_bivariate_lee_l")

    def createInstance(self):
        return GlobalLeesLAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Tests whether two fields are spatially co-clustered using Lee's "
            "(2001) global L statistic - a single study-area-wide summary "
            "of the same spatial smoothing of Pearson's r that the "
            "plugin's existing Local Bivariate Lee's L computes feature by "
            "feature. This is the Global/Local pairing Global Moran's I has "
            "to Local Moran's I, applied to a two-variable relationship "
            "instead of one.\n\n"
            "Global L answers a different question than a plain Pearson "
            "correlation: it asks whether X at one location predicts Y at "
            "NEARBY locations, not just Y at the same location. Two fields "
            "can be spatially co-clustered (high Global L) even with a weak "
            "or zero simple correlation, if their relationship only emerges "
            "once neighboring values are pooled - and conversely a strong "
            "simple correlation can coexist with a low Global L if the "
            "co-clustering has no spatial coherence.\n\n"
            "Under this plugin's row-standardized weights convention, "
            "Global L is mathematically exactly the mean of the local l_i "
            "values - run Local Bivariate Lee's L afterward on the same "
            "field pair to see WHERE the global co-clustering concentrates. "
            "Significance is assessed by permutation (999 shuffles by "
            "default, seed 42), holding the spatial structure and Field X "
            "fixed while reshuffling Field Y."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_X,
                "Field X",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_Y,
                "Field Y",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERMUTATIONS,
                "Number of permutations (Monte Carlo)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=999,
                minValue=99,
                maxValue=9999,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANDOM_SEED,
                "Random seed (reproducibility)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=42,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Global Bivariate Lee's L diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_x_name = self.parameterAsString(parameters, self.FIELD_X, context)
        field_y_name = self.parameterAsString(parameters, self.FIELD_Y, context)
        if field_x_name == field_y_name:
            raise QgsProcessingException("Field X and Field Y must be different fields.")

        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "global_lee_l_report.html")

        feedback.pushInfo("Generating spatial weights matrix...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source, weight_type, k_neighbors=k_neighbors, distance_band=distance_band, feedback=feedback
        )
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Extracting field values...")
        x_dict, y_dict = {}, {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            xv = f.attribute(field_x_name)
            yv = f.attribute(field_y_name)
            if xv is None or xv == NULL or str(xv) == "NULL":
                continue
            if yv is None or yv == NULL or str(yv) == "NULL":
                continue
            try:
                x_dict[f.id()] = float(xv)
                y_dict[f.id()] = float(yv)
            except (ValueError, TypeError):
                continue

        valid_id_order = [fid for fid in id_order if fid in x_dict and fid in y_dict]
        x_arr = np.array([x_dict[fid] for fid in valid_id_order])
        y_arr = np.array([y_dict[fid] for fid in valid_id_order])

        numeric_summary = numeric_quality_summary(source.featureCount(), {fid: 1 for fid in valid_id_order}, x_arr)
        neighborhood_summary = neighbor_summary(neighbors, valid_id_order)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, neighborhood_summary, crs_warning)

        if len(valid_id_order) <= 3:
            raise QgsProcessingException("At least 4 valid features with both fields populated are required.")
        if len(set(x_arr)) == 1 or len(set(y_arr)) == 1:
            raise QgsProcessingException("Both fields must have variation; a constant field cannot be spatially co-clustered.")

        feedback.pushInfo(f"Calculating Global Bivariate Lee's L using {permutations} permutations...")
        result = calculate_global_lee_l(x_arr, y_arr, neighbors, weights, valid_id_order, permutations=permutations, seed=seed)

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"Global L = {result['observed_l']:.4f}, z = {result['z_score']:.3f}, p = {result['p_value']:.4f}"
        )

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_x_name, field_y_name, len(valid_id_order), result,
            numeric_summary, neighborhood_summary, crs_warning, weight_type, permutations,
        )

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(
        self, path, field_x, field_y, n, result, numeric_summary, neighborhood_summary, crs_warning, weight_type, permutations,
    ):
        l_val = result["observed_l"]
        z = result["z_score"]
        p = result["p_value"]
        is_significant = p < 0.05

        if is_significant and l_val > 0:
            pattern = "Positive Spatial Co-Clustering"
            color = "#e31a1c"
            desc = f"Global L = {l_val:.4f} is significantly positive (z = {z:.2f}, p = {p:.4f}): high values of one field tend to be near high values of the other (and lows near lows)."
        elif is_significant and l_val < 0:
            pattern = "Negative Spatial Co-Clustering"
            color = "#1f78b4"
            desc = f"Global L = {l_val:.4f} is significantly negative (z = {z:.2f}, p = {p:.4f}): high values of one field tend to be near low values of the other."
        else:
            pattern = "No Significant Spatial Co-Clustering"
            color = "#718096"
            desc = f"Global L = {l_val:.4f} is not distinguishable from zero (p = {p:.4f}): no evidence the two fields are spatially co-clustered."

        if neighborhood_summary["isolated"] > 0:
            next_action = "Increase the neighborhood size before interpreting Global L; isolated features are excluded from the spatial smoothing."
        elif is_significant:
            next_action = f"Run Local Bivariate Lee's L on the same field pair ({html.escape(field_x)} / {html.escape(field_y)}) to map where the co-clustering concentrates."
        else:
            next_action = "Check a plain Pearson correlation as a sanity comparison; a non-spatial relationship with no spatial coherence will show up there but not here."

        guidance_html = analyst_guidance_html(
            "Global Bivariate Lee's L",
            "Global Bivariate Lee's L tests whether two fields are spatially co-clustered across the whole study area, complementing the plugin's existing Local Bivariate Lee's L.",
            [
                "Both fields are measured on a scale where spatial smoothing (neighbor averaging) is meaningful.",
                "The relationship is genuinely spatial in nature, not just a same-location correlation.",
                "The neighborhood graph is not dominated by isolated features.",
            ],
            [
                "Treating Global L as a replacement for Pearson correlation - they answer different questions and can disagree.",
                "Interpreting a significant Global L as proof of causation between the two fields.",
                "Skipping Local Bivariate Lee's L after a significant Global L - the global figure hides which part of the study area drives it.",
            ],
            [
                "Local Bivariate Lee's L to map where co-clustering concentrates",
                "Global Moran's I on each field individually to check they are each spatially structured to begin with",
                "Bivariate Choropleth Map for a visual first look at the same relationship",
            ],
            "Run Global L before Local L as a screening step - a non-significant Global L means local co-clustering, if any, is likely too weak or scattered to interpret with confidence.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Global Bivariate Lee's L Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 760px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    .interpretation-box {{ background-color: #f8fafc; border-left: 5px solid {color}; padding: 20px; border-radius: 4px; margin-bottom: 30px; }}
    .status-title {{ font-size: 1.3rem; font-weight: 800; color: {color}; margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.05em; }}
    .status-desc {{ color: #4a5568; font-size: 0.95rem; margin: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 0.9rem; }}
    th {{ background-color: #ebf8ff; color: #2b6cb0; font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    .metric-name {{ font-weight: 600; color: #2d3748; }}
    .metric-val {{ font-family: monospace; font-size: 1rem; font-weight: 600; }}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
    section {{ margin: 28px 0; }}
    h2 {{ color: #1a202c; font-size: 1.15rem; margin: 0 0 12px 0; }}
    .next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
    {analyst_guidance_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Global Bivariate Lee's L</h1>
        <p class="subtitle">Field X: <strong>{html.escape(field_x)}</strong> | Field Y: <strong>{html.escape(field_y)}</strong> | N: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_type)}</strong> | Permutations: <strong>{permutations}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">{pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>Global Bivariate Lee's L (Lee, 2001) is a single study-area-wide measure of spatial co-clustering between two fields - a spatial smoothing of Pearson's r. Under row-standardized weights it is mathematically the mean of the local l_i values computed by Local Bivariate Lee's L. Significance is assessed via {permutations} random permutations.</p>
    </section>

    <table>
        <thead><tr><th>Global Lee's L Diagnostic</th><th>Statistical Value</th></tr></thead>
        <tbody>
            <tr><td class="metric-name">Observed Global L</td><td class="metric-val">{l_val:.6f}</td></tr>
            <tr><td class="metric-name">Permuted Mean</td><td class="metric-val">{result['permuted_mean']:.6f}</td></tr>
            <tr><td class="metric-name">Permuted Std. Dev.</td><td class="metric-val">{result['permuted_std']:.6f}</td></tr>
            <tr><td class="metric-name">z-score</td><td class="metric-val">{z:.6f}</td></tr>
            <tr><td class="metric-name">p-value</td><td class="metric-val">{p:.6f}</td></tr>
        </tbody>
    </table>

    {diagnostics_html(numeric_summary, neighborhood_summary, crs_warning)}

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>

    {caveats_html("Global Bivariate Lee's L", neighborhood_summary, numeric_summary)}

    {guidance_html}

    <footer>
        Generated by PlanX GeoStats Lab global spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
