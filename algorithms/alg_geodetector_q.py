# -*- coding: utf-8 -*-
"""Geodetector Q-Statistic (Spatial Stratified Heterogeneity) Processing Algorithm."""
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

from ..core.advanced_stats_engines import calculate_geodetector_q, bin_into_quantiles
from ..core.reporting import analyst_guidance_css, analyst_guidance_html

from ._icons import algorithm_icon


class GeodetectorQAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD_Y = "FIELD_Y"
    STRATA_MODE = "STRATA_MODE"
    FIELD_STRATA = "FIELD_STRATA"
    N_BINS = "N_BINS"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "geodetector_q_statistic"

    def displayName(self) -> str:
        return "Geodetector Q-Statistic (Spatial Stratified Heterogeneity)"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def icon(self):
        return algorithm_icon("geodetector_q_statistic")

    def createInstance(self):
        return GeodetectorQAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Tests spatial stratified heterogeneity: whether a categorical "
            "zoning/district/classification field explains the variance of "
            "a continuous outcome field (Wang et al., 2010, IJGIS; Wang, "
            "Zhang & Fu, 2016, Ecological Indicators). This is a "
            "fundamentally different paradigm from every other tool in "
            "this plugin - no spatial weights matrix is built at all. "
            "Instead of asking 'are nearby features similar' (autocorrelation), "
            "Geodetector asks 'does this partition of the study area into "
            "categories explain the variation I see' (stratification).\n\n"
            "The Q-statistic ranges from 0 (the stratification explains "
            "nothing - within-stratum variance is as large as total "
            "variance) to 1 (the stratification explains everything - each "
            "stratum is internally uniform). Formally:\n"
            "  q = 1 - [sum_h(N_h * var_h)] / [N * var_total]\n"
            "where h indexes strata, N_h is stratum size, and var_h is the "
            "within-stratum variance.\n\n"
            "If the categorical field you want to test already exists "
            "(zoning code, district ID), use it directly. If you only have "
            "a continuous field you want to test as a stratification "
            "driver (e.g. does distance-to-center explain accessibility "
            "variance), this tool can quantile-bin it into strata first. "
            "Significance is assessed by permutation (999 shuffles of "
            "strata labels by default, seed 42) rather than the classical "
            "F-approximation, which assumes no within-stratum spatial "
            "autocorrelation - an assumption that rarely holds for "
            "planning data. This is the single-factor q-statistic; the "
            "full Geodetector framework's interaction/risk/ecological "
            "detectors are a natural extension for a future version."
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
                self.FIELD_Y,
                "Outcome field (continuous, to explain)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.STRATA_MODE,
                "Stratification source",
                options=["Use an existing categorical field", "Quantile-bin a continuous field"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_STRATA,
                "Stratification field (categorical field, or continuous field to bin)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_BINS,
                "Number of quantile bins (Quantile-bin mode only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=2,
                maxValue=20,
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
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Geodetector Q-Statistic diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_y_name = self.parameterAsString(parameters, self.FIELD_Y, context)
        field_strata_name = self.parameterAsString(parameters, self.FIELD_STRATA, context)
        strata_mode = self.parameterAsEnum(parameters, self.STRATA_MODE, context)
        n_bins = self.parameterAsInt(parameters, self.N_BINS, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)

        if field_y_name == field_strata_name:
            raise QgsProcessingException("Outcome field and stratification field must be different fields.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "geodetector_q_report.html")

        feedback.pushInfo("Extracting field values...")
        y_dict, strata_raw = {}, {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            yv = f.attribute(field_y_name)
            sv = f.attribute(field_strata_name)
            if yv is None or yv == NULL or str(yv) == "NULL":
                continue
            if sv is None or sv == NULL or str(sv) == "NULL":
                continue
            try:
                y_dict[f.id()] = float(yv)
            except (ValueError, TypeError):
                continue
            strata_raw[f.id()] = sv

        valid_ids = [fid for fid in y_dict if fid in strata_raw]
        if len(valid_ids) <= 3:
            raise QgsProcessingException("At least 4 valid features with both fields populated are required.")

        y = np.array([y_dict[fid] for fid in valid_ids])

        if strata_mode == 1:
            try:
                strata_vals = np.array([float(strata_raw[fid]) for fid in valid_ids])
            except (ValueError, TypeError):
                raise QgsProcessingException("Quantile-bin mode requires a numeric stratification field.")
            feedback.pushInfo(f"Quantile-binning stratification field into {n_bins} bins...")
            strata = bin_into_quantiles(strata_vals, n_bins)
            strata_desc = f"{n_bins} quantile bins of '{field_strata_name}'"
        else:
            unique_vals = sorted(set(str(strata_raw[fid]) for fid in valid_ids))
            if len(unique_vals) < 2:
                raise QgsProcessingException("Stratification field must have at least 2 distinct categories.")
            if len(unique_vals) > 30:
                feedback.pushWarning(
                    f"Stratification field has {len(unique_vals)} distinct categories; "
                    "Q-statistic strata should typically be a coarse classification, not a near-unique ID."
                )
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            strata = np.array([val_to_idx[str(strata_raw[fid])] for fid in valid_ids])
            strata_desc = f"'{field_strata_name}' ({len(unique_vals)} categories)"

        if len(set(y)) == 1:
            raise QgsProcessingException("Outcome field must have variation; a constant field has no variance to explain.")

        feedback.pushInfo(f"Calculating Geodetector Q-statistic using {permutations} permutations...")
        result = calculate_geodetector_q(y, strata, permutations=permutations, seed=seed)

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"Q = {result['q_statistic']:.4f} across {result['n_strata']} strata, p = {result['p_value']:.4f}"
        )

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_y_name, strata_desc, len(valid_ids), result, permutations,
        )

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(self, path, field_y, strata_desc, n, result, permutations):
        q = result["q_statistic"]
        p = result["p_value"]
        n_strata = result["n_strata"]
        is_significant = p < 0.05

        if is_significant and q >= 0.5:
            pattern = "Strong Spatial Stratified Heterogeneity"
            color = "#e31a1c"
            desc = f"Q = {q:.4f} (p = {p:.4f}): the stratification explains a large share of the variance in {html.escape(field_y)}."
        elif is_significant:
            pattern = "Significant but Modest Stratified Heterogeneity"
            color = "#dd6b20"
            desc = f"Q = {q:.4f} (p = {p:.4f}): the stratification explains a statistically detectable but modest share of variance."
        else:
            pattern = "No Significant Stratified Heterogeneity"
            color = "#718096"
            desc = f"Q = {q:.4f} (p = {p:.4f}): this stratification does not explain variance in {html.escape(field_y)} beyond what random grouping would."

        if is_significant and q >= 0.5:
            next_action = "Report the per-stratum means/variances below to identify which categories drive the outcome; consider this stratification as a candidate explanatory factor in Multiple Regression or GWR."
        elif is_significant:
            next_action = "This stratification has some explanatory value but likely is not the dominant driver; test alternative stratifications or combine with continuous predictors in a regression model."
        else:
            next_action = "Try a different stratification (finer/coarser bins, or a different categorical field) before concluding the outcome field has no spatial stratified structure."

        strata_rows = ""
        for stratum in result["per_stratum"]:
            variance = stratum["std"] ** 2
            strata_rows += (
                f"<tr><td class='metric-name'>Stratum {stratum['stratum']}</td>"
                f"<td class='metric-val'>{stratum['n']}</td>"
                f"<td class='metric-val'>{stratum['mean']:.4f}</td>"
                f"<td class='metric-val'>{variance:.4f}</td></tr>\n"
            )

        guidance_html = analyst_guidance_html(
            "Geodetector Q-Statistic",
            "The Geodetector Q-statistic tests spatial stratified heterogeneity - whether a categorical partition of the study area explains variance in a continuous outcome - a different paradigm from weights-based spatial autocorrelation.",
            [
                "Strata are a meaningful planning classification (zoning, district), not an arbitrary or near-unique split.",
                "Each stratum has enough features for its variance to be a stable estimate.",
                "The outcome field genuinely varies across, not just within, the proposed strata.",
            ],
            [
                "Using Q as a substitute for spatial autocorrelation tests - a high Q says the categories differ, not that nearby features are similar.",
                "Testing a near-unique ID field as strata (e.g. parcel ID) - guaranteed to inflate Q without meaning.",
                "Comparing Q across stratifications with very different numbers of categories without checking per-stratum sample sizes.",
            ],
            [
                "Multiple Regression Analysis to combine this stratification with continuous predictors",
                "Data Readiness Audit to check per-stratum sample sizes before trusting the variance estimates",
                "Local Moran's I / Getis-Ord Gi* if the real question is proximity-based clustering, not category-based partitioning",
            ],
            "Use Geodetector Q to screen candidate categorical explanatory factors (zoning, land-use class) before committing to them in a full regression model - a low Q here means that factor alone is unlikely to matter much as a predictor.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Geodetector Q-Statistic Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 800px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
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
    .scroll-table {{ overflow-x: auto; }}
    {analyst_guidance_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Geodetector Q-Statistic (Spatial Stratified Heterogeneity)</h1>
        <p class="subtitle">Outcome: <strong>{html.escape(field_y)}</strong> | Strata: <strong>{html.escape(strata_desc)}</strong> | N: <strong>{n}</strong> | Permutations: <strong>{permutations}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">{pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>The Geodetector Q-statistic (Wang et al., 2010) measures how much of the variance in an outcome field is explained by a categorical stratification, ranging from 0 (no explanatory power) to 1 (strata perfectly separate the outcome). No spatial weights matrix is used; significance is assessed via {permutations} random permutations of strata labels.</p>
    </section>

    <table>
        <thead><tr><th>Q-Statistic Diagnostic</th><th>Statistical Value</th></tr></thead>
        <tbody>
            <tr><td class="metric-name">Observed Q</td><td class="metric-val">{q:.6f}</td></tr>
            <tr><td class="metric-name">Number of Strata</td><td class="metric-val">{n_strata}</td></tr>
            <tr><td class="metric-name">Permuted Mean</td><td class="metric-val">{result['permuted_mean']:.6f}</td></tr>
            <tr><td class="metric-name">Permuted Std. Dev.</td><td class="metric-val">{result['permuted_std']:.6f}</td></tr>
            <tr><td class="metric-name">z-score</td><td class="metric-val">{result['z_score']:.6f}</td></tr>
            <tr><td class="metric-name">p-value (one-tailed)</td><td class="metric-val">{p:.6f}</td></tr>
        </tbody>
    </table>

    <section>
        <h2>Per-Stratum Breakdown</h2>
        <div class="scroll-table">
        <table>
            <thead><tr><th>Stratum</th><th>N</th><th>Mean</th><th>Variance</th></tr></thead>
            <tbody>
                {strata_rows}
            </tbody>
        </table>
        </div>
    </section>

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>

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
