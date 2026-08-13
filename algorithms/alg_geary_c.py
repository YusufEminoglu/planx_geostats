# -*- coding: utf-8 -*-
"""Geary's C (Spatial Autocorrelation) Processing Algorithm."""
from __future__ import annotations

import logging
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
from ..core.advanced_stats_engines import calculate_geary_c
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


logger = logging.getLogger("PlanX GeoStats Lab")


class GearyCAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "geary_c_autocorrelation"

    def displayName(self) -> str:
        return "Geary's C (Spatial Autocorrelation)"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def icon(self):
        return algorithm_icon("geary_c_autocorrelation")

    def createInstance(self):
        return GearyCAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Tests spatial autocorrelation using Geary's C (Geary, 1954; "
            "Cliff & Ord, 1981), the sum-of-squared-differences alternative "
            "to Global Moran's I. Unlike Moran's I, Geary's C uses the "
            "opposite direction convention: C < 1 means similar values are "
            "spatially clustered (positive autocorrelation), C > 1 means "
            "dissimilar values are adjacent (negative autocorrelation), and "
            "C = 1 is the no-autocorrelation reference point. Because it "
            "compares each pair of neighbors directly rather than each "
            "value to the global mean, Geary's C is more sensitive to "
            "local dissimilarity than Moran's I, which can make it pick "
            "up short-range structure Moran's I averages away.\n\n"
            "Significance is assessed by permutation (999 shuffles by "
            "default, seed 42 for reproducibility) rather than the exact "
            "Cliff-Ord randomization variance formula, avoiding that "
            "formula's own distributional assumptions. Report fields: "
            "observed_c, expected_c (= 1.0 exactly), permuted_mean, "
            "permuted_std, z_score, p_value.\n\n"
            "Run this alongside Global Moran's I rather than instead of "
            "it: the two statistics can disagree - Geary's C can flag "
            "significant local dissimilarity in a field where Moran's I "
            "sees no overall clustering, since Moran's I is dominated by "
            "how far each value sits from the global mean while Geary's C "
            "only looks at neighbor-to-neighbor differences. A geographic "
            "CRS or a distance-band threshold chosen without checking "
            "Incremental Spatial Autocorrelation first will bias both "
            "statistics identically to Moran's I - the same weight-choice "
            "cautions apply here."
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
                self.FIELD,
                "Target numeric field to analyze",
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
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Geary's C diagnostic report"))

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
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "geary_c_report.html")

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        feedback.pushInfo("Generating spatial weights matrix...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source, weight_type, k_neighbors=k_neighbors, distance_band=distance_band, feedback=feedback
        )
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Extracting target field values...")
        y_dict = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val = f.attribute(field_name)
            if val is None or val == NULL or str(val) == "NULL":
                continue
            try:
                y_dict[f.id()] = float(val)
            except (ValueError, TypeError):
                continue

        valid_id_order = [fid for fid in id_order if fid in y_dict]
        y = np.array([y_dict[fid] for fid in valid_id_order])
        numeric_summary = numeric_quality_summary(source.featureCount(), y_dict, y)
        neighborhood_summary = neighbor_summary(neighbors, valid_id_order)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, neighborhood_summary, crs_warning)

        if len(y) <= 3:
            raise QgsProcessingException("At least 4 valid features with numeric values are required for Geary's C analysis.")
        if numeric_summary["is_constant"]:
            raise QgsProcessingException("Geary's C requires variation in the target field; all valid values are identical.")

        feedback.pushInfo(f"Calculating Geary's C using {permutations} permutations...")
        result = calculate_geary_c(y, neighbors, weights, valid_id_order, permutations=permutations, seed=seed)

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"Geary's C = {result['observed_c']:.4f} (expected 1.0), z = {result['z_score']:.3f}, p = {result['p_value']:.4f}"
        )

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, len(y), result, numeric_summary, neighborhood_summary, crs_warning, weight_type, permutations,
        )

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(
        self, path, field_name, n, result, numeric_summary, neighborhood_summary, crs_warning, weight_type, permutations,
    ):
        c = result["observed_c"]
        z = result["z_score"]
        p = result["p_value"]
        is_significant = p < 0.05

        if is_significant and c < 1.0:
            pattern = "Positive Spatial Autocorrelation (Clustered)"
            color = "#e31a1c"
            desc = f"Geary's C = {c:.4f} is significantly below 1.0 (z = {z:.2f}, p = {p:.4f}): similar values are spatially clustered."
        elif is_significant and c > 1.0:
            pattern = "Negative Spatial Autocorrelation (Dispersed)"
            color = "#1f78b4"
            desc = f"Geary's C = {c:.4f} is significantly above 1.0 (z = {z:.2f}, p = {p:.4f}): dissimilar values are adjacent more than expected."
        else:
            pattern = "Random (No Significant Autocorrelation)"
            color = "#718096"
            desc = f"Geary's C = {c:.4f} is not distinguishable from the expected value of 1.0 (p = {p:.4f})."

        if neighborhood_summary["isolated"] > 0:
            next_action = "Increase the neighborhood size before interpreting Geary's C; isolated features contribute nothing to the statistic."
        elif is_significant and c < 1.0:
            next_action = "Run Local Moran's I or Getis-Ord Gi* to locate where the clustering occurs."
        elif is_significant:
            next_action = "Review whether the dispersion reflects a real repulsion process or a data-coding artifact (e.g. alternating administrative categories)."
        else:
            next_action = "Compare against Global Moran's I; a disagreement between the two statistics is itself informative about the scale of the pattern."

        guidance_html = analyst_guidance_html(
            "Geary's C",
            "Geary's C tests spatial autocorrelation via neighbor-to-neighbor squared differences, complementing Global Moran's I with a statistic more sensitive to local dissimilarity.",
            [
                "The input layer uses a projected CRS when distance bands are involved.",
                "The analysis field has enough numeric variation and few skipped records.",
                "The neighborhood graph is neither dominated by isolated features nor fully connected.",
            ],
            [
                "Treating Geary's C and Moran's I as redundant - they can disagree, and the disagreement is informative.",
                "Forgetting the reversed direction convention: C < 1 is clustering, not C > 1.",
                "Using too few permutations for a stable p-value near a decision threshold.",
            ],
            [
                "Global Moran's I for the standard-convention companion statistic",
                "Local Moran's I / Getis-Ord Gi* to locate significant clustering",
                "Join Count Statistics if the field is binary/categorical rather than continuous",
            ],
            "Use Geary's C as a second opinion on Global Moran's I, not a replacement - agreement between the two increases confidence, disagreement flags a scale-dependent or locally-driven pattern worth local investigation.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Geary's C Report</title>
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
        <h1>Geary's C (Spatial Autocorrelation)</h1>
        <p class="subtitle">Field Analyzed: <strong>{html.escape(field_name)}</strong> | Feature Count: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_type)}</strong> | Permutations: <strong>{permutations}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">Spatial Pattern: {pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>Geary's C tests spatial autocorrelation via the sum of squared differences between neighbors, using the opposite direction convention from Moran's I: values below 1.0 indicate clustering, values above 1.0 indicate dispersion. Significance is assessed empirically via {permutations} random permutations rather than an asymptotic formula.</p>
    </section>

    <table>
        <thead><tr><th>Geary's C Diagnostic</th><th>Statistical Value</th></tr></thead>
        <tbody>
            <tr><td class="metric-name">Observed Geary's C</td><td class="metric-val">{c:.6f}</td></tr>
            <tr><td class="metric-name">Expected C (no autocorrelation)</td><td class="metric-val">1.000000</td></tr>
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

    {caveats_html("Geary's C", neighborhood_summary, numeric_summary)}

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
