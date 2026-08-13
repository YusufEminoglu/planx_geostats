# -*- coding: utf-8 -*-
"""Lagrange Multiplier Diagnostics Processing Algorithm."""
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
from ..core.advanced_stats_engines import calculate_lm_diagnostics
from ..core.analysis_diagnostics import regression_quality_html, regression_quality_summary
from ..core.reporting import analyst_guidance_css, analyst_guidance_html

from ._icons import algorithm_icon


class LMDiagnosticsAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "lm_diagnostics"

    def displayName(self) -> str:
        return "Lagrange Multiplier Diagnostics (Spatial Model Selection)"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("lm_diagnostics")

    def createInstance(self):
        return LMDiagnosticsAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Runs the formal Lagrange Multiplier test battery (Anselin, "
            "Bera, Florax & Yoon, 1996) that this plugin's own OLS, Spatial "
            "Autoregression, and Spatial Error Regression tools already "
            "tell users to 'consult' before choosing between a Spatial Lag "
            "(SAR) and Spatial Error (SEM) specification - without actually "
            "providing it until now. Fits OLS internally, then computes "
            "four chi-square(1) test statistics: LM-lag and LM-error (the "
            "classic tests, each ignoring the other form of dependence), "
            "and Robust LM-lag / Robust LM-error (each corrected for the "
            "presence of the other).\n\n"
            "Decision rule (Anselin's standard classification): if only "
            "LM-lag is significant, use Spatial Lag/Durbin; if only "
            "LM-error is significant, use Spatial Error; if BOTH are "
            "significant, consult the ROBUST versions - whichever robust "
            "test survives indicates the correct specification; if both "
            "robust tests are significant, a model combining both spatial "
            "lag and error components is indicated (e.g. Spatial Durbin); "
            "if neither classic test is significant, OLS is adequate and "
            "no spatial specification is needed. The report states this "
            "recommendation explicitly, along with the joint SARMA "
            "chi-square(2) statistic (LM-lag + Robust LM-error, which by "
            "construction equals LM-error + Robust LM-lag) as a combined "
            "check.\n\n"
            "Run this BEFORE Spatial Autoregression or Spatial Error "
            "Regression, not after - it is specifically designed to answer "
            "the question those two tools' own documentation defers."
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
                self.DEP_VAR,
                "Dependent variable field",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.INDEPENDENTS,
                "Independent variable fields (select one or more)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True,
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
                defaultValue=8,
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
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Lagrange Multiplier diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        indep_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)
        if not indep_fields:
            raise QgsProcessingException("At least one independent variable must be selected.")

        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]
        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "lm_diagnostics_report.html")

        dep_idx = source.fields().lookupField(dep_var)
        indep_idxs = [source.fields().lookupField(name) for name in indep_fields]
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        for name, idx in zip(indep_fields, indep_idxs):
            if idx < 0:
                raise QgsProcessingException(f"Independent field '{name}' not found.")

        feedback.pushInfo("Extracting numeric values and filtering missing data...")
        dep_vals, indep_vals, valid_fids = [], [], []
        skipped = 0
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            y_val = f.attribute(dep_idx)
            if y_val is None or y_val == NULL or str(y_val) == "NULL":
                skipped += 1
                continue
            has_null = False
            f_indeps = []
            for i_idx in indep_idxs:
                x_val = f.attribute(i_idx)
                if x_val is None or x_val == NULL or str(x_val) == "NULL":
                    has_null = True
                    break
                try:
                    f_indeps.append(float(x_val))
                except (ValueError, TypeError):
                    has_null = True
                    break
            if has_null:
                skipped += 1
                continue
            try:
                dep_vals.append(float(y_val))
                indep_vals.append(f_indeps)
                valid_fids.append(f.id())
            except (ValueError, TypeError):
                skipped += 1
                continue
            feedback.setProgress(int(20 * (idx / total)))

        n = len(dep_vals)
        p = len(indep_fields)
        if n <= p + 1:
            raise QgsProcessingException(
                f"Insufficient valid observations ({n}). Must be greater than the number of independent variables ({p}) + 1."
            )

        y = np.array(dep_vals)
        X_data = np.array(indep_vals)
        model_quality = regression_quality_summary(y, X_data, indep_fields, source.featureCount())
        feedback.pushInfo(
            "Model quality diagnostics: "
            f"{model_quality['used_records']} complete record(s), "
            f"{model_quality['skipped_records']} skipped record(s), "
            f"{model_quality['predictor_count']} predictor(s)."
        )
        for risk in model_quality["risks"]:
            feedback.pushWarning(risk)

        feedback.pushInfo("Generating spatial weights matrix...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source, weight_type, k_neighbors=k_neighbors, distance_band=distance_band, feedback=feedback
        )
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Fitting OLS baseline and computing LM diagnostics...")
        result = calculate_lm_diagnostics(y, X_data, neighbors, weights, valid_fids)

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"LM-lag={result['lm_lag']:.4f} (p={result['lm_lag_p']:.4f}), "
            f"LM-error={result['lm_error']:.4f} (p={result['lm_error_p']:.4f}), "
            f"Robust LM-lag={result['rlm_lag']:.4f} (p={result['rlm_lag_p']:.4f}), "
            f"Robust LM-error={result['rlm_error']:.4f} (p={result['rlm_error_p']:.4f})"
        )
        feedback.pushInfo(result["recommendation"])

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(html_path, dep_var, n, weight_type, result, model_quality)

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(self, path, dep_var, n, weight_type, result, model_quality):
        def status_row(label, stat, p, sig_text, nonsig_text):
            sig = p < 0.05
            badge = (
                f'<span class="badge badge-danger">{html.escape(sig_text)}</span>'
                if sig
                else f'<span class="badge badge-success">{html.escape(nonsig_text)}</span>'
            )
            return f"<tr><td><strong>{label}</strong></td><td>{stat:.4f}</td><td>{p:.6f}</td><td>{badge}</td></tr>"

        rows = (
            status_row("LM-lag", result["lm_lag"], result["lm_lag_p"], "Significant", "Not Significant")
            + status_row("LM-error", result["lm_error"], result["lm_error_p"], "Significant", "Not Significant")
            + status_row("Robust LM-lag", result["rlm_lag"], result["rlm_lag_p"], "Significant", "Not Significant")
            + status_row("Robust LM-error", result["rlm_error"], result["rlm_error_p"], "Significant", "Not Significant")
            + status_row("Joint SARMA (chi2, 2 df)", result["sarma"], result["sarma_p"], "Significant", "Not Significant")
        )

        guidance_html = analyst_guidance_html(
            "Lagrange Multiplier Diagnostics",
            "The LM test battery formally determines whether a Spatial Lag, Spatial Error, or combined specification is indicated by residual spatial dependence patterns after OLS.",
            [
                "The OLS baseline itself is reasonably specified (check the model-quality diagnostics below).",
                "The chosen spatial weights type/scale genuinely reflects the process being tested.",
                "Both classic and robust statistics are read together, not classic tests in isolation.",
            ],
            [
                "Choosing a spatial specification from the classic LM-lag/LM-error tests alone when both are significant - always check the robust versions in that case.",
                "Running SAR or SEM directly without ever running this diagnostic first.",
                "Treating a significant LM-error as automatic proof of a genuine spatial process rather than possible omitted-variable nuisance autocorrelation.",
            ],
            [
                "Spatial Autoregression (SAR) if a lag specification is indicated",
                "Spatial Error Regression (SEM) if an error specification is indicated",
                "Spatial Durbin Model (SDM) if both robust tests are significant",
            ],
            "Use this diagnostic as the formal gate between OLS and a spatial model family - the plugin's own SAR/SEM tools reference this exact test without providing it, so run this first and cite the recommendation when justifying the model choice.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Lagrange Multiplier Diagnostics Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 860px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    .interpretation-box {{ background-color: #f8fafc; border-left: 5px solid #3182ce; padding: 20px; border-radius: 4px; margin-bottom: 30px; }}
    .status-title {{ font-size: 1.1rem; font-weight: 800; color: #2b6cb0; margin: 0 0 10px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 0.9rem; }}
    th {{ background-color: #ebf8ff; color: #2b6cb0; font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    .badge {{ display: inline-block; padding: 4px 8px; font-size: 0.75rem; font-weight: 700; border-radius: 4px; }}
    .badge-success {{ background-color: #c6f6d5; color: #22543d; }}
    .badge-danger {{ background-color: #fed7d7; color: #742a2a; }}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
    {analyst_guidance_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Lagrange Multiplier Diagnostics</h1>
        <p class="subtitle">Dependent Variable: <strong>{html.escape(dep_var)}</strong> | N: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_type)}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">Recommendation</h2>
        <p>{html.escape(result['recommendation'])}</p>
    </div>

    <h2>LM Test Battery</h2>
    <table>
        <thead><tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Status</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>

    {regression_quality_html(model_quality)}

    {guidance_html}

    <footer>
        Generated by PlanX GeoStats Lab spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
