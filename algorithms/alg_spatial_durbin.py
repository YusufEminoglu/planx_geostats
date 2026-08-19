# -*- coding: utf-8 -*-
"""Spatial Durbin Model (SDM) Processing Algorithm."""
from __future__ import annotations

import math
import os
import tempfile
import html
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsProject,
    QgsFeature,
    QgsField,
    QgsSymbol,
    QgsRendererRange,
    QgsGraduatedSymbolRenderer,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFeatureSink,
    QgsProcessingOutputHtml,
    QgsFeatureSink,
)

from ..core.weights import build_weights_matrix
from ..core.advanced_stats_engines import calculate_spatial_durbin
from ..core.analysis_diagnostics import regression_quality_html, regression_quality_summary
from ..core.layer_metadata import apply_output_metadata
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg

from ._icons import algorithm_icon


class SpatialDurbinAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "spatial_durbin_model"

    def displayName(self) -> str:
        return "Spatial Durbin Model (SDM)"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("spatial_durbin_model")

    def createInstance(self):
        return SpatialDurbinAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Estimates y = rho*Wy + X*beta + WX*theta + e - the Spatial "
            "Durbin Model (LeSage & Pace, 2009), which lags BOTH the "
            "dependent variable and every explanatory variable. SDM nests "
            "Spatial Lag (SAR) as the restricted case theta=0, and a "
            "restricted form of Spatial Error (SEM) as the case theta = "
            "-rho*beta - making it the natural model to try when Lagrange "
            "Multiplier Diagnostics flags both robust LM-lag AND robust "
            "LM-error as significant, since a pure SAR or SEM specification "
            "won't fully resolve dependence that has both components.\n\n"
            "Estimated by Spatial Two-Stage Least Squares (Kelejian & "
            "Prucha, 1998), NOT Maximum Likelihood: Wy is endogenous (the "
            "simultaneous spatial feedback correlates it with the error "
            "term), so plain OLS on this specification is inconsistent. "
            "The engine instruments Wy with [WX, W^2X] - the standard "
            "instrument set for the spatial lag model family - rather than "
            "computing an eigenvalue-based log-determinant Jacobian term, "
            "trading a small amount of asymptotic efficiency for a much "
            "smaller, closed-form correctness surface with no optional "
            "dependency required.\n\n"
            "Output fields: residual and fitted (predicted) values per "
            "feature, and an HTML report with all coefficients (Intercept, "
            "X variables, their spatial lags WX, and rho). A key "
            "interpretation trap in SDM: raw beta coefficients are NOT "
            "directly interpretable as marginal effects, because rho "
            "creates feedback loops through neighboring locations - "
            "LeSage and Pace's direct/indirect/total impact decomposition "
            "is the theoretically correct way to interpret SDM coefficients "
            "and is a natural extension for a future version; treat the "
            "raw beta/theta table here as a specification and significance "
            "screen, not a final marginal-effects report."
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
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output residuals layer",
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output SDM HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Spatial Durbin Model diagnostic report"))

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
            html_path = os.path.join(tempfile.gettempdir(), "spatial_durbin_report.html")

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
        if n <= 2 * p + 2:
            raise QgsProcessingException(
                f"Insufficient valid observations ({n}) for a Spatial Durbin Model with {p} predictor(s)."
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

        feedback.pushInfo("Estimating Spatial Durbin Model via Spatial 2SLS...")
        result = calculate_spatial_durbin(y, X_data, neighbors, weights, valid_fids, indep_fields)
        feedback.pushInfo(f"Model R2={result['r2']:.4f}, rho={result['rho']:.4f} (p={result['rho_p']:.4f})")

        if feedback.isCanceled():
            return {}

        out_fields = source.fields()
        out_fields.append(QgsField("residual", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("fitted", QVariant.Double, len=12, prec=6))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs()
        )
        self.out_layer_id = dest_id

        residuals = result["residuals"]
        fitted = y - residuals
        results_map = {fid: (residuals[i], fitted[i]) for i, fid in enumerate(valid_fids)}

        feedback.pushInfo("Writing residual attributes...")
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)
            fid = f.id()
            if fid in results_map:
                res, fit = results_map[fid]
                out_feat.setAttribute("residual", float(res))
                out_feat.setAttribute("fitted", float(fit))
            else:
                out_feat.setAttribute("residual", None)
                out_feat.setAttribute("fitted", None)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 30 * (current / total)))

        feedback.pushInfo("Generating SDM HTML diagnostics report...")
        self.write_html_report(html_path, dep_var, weight_type, result, model_quality)

        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(self, path, dep_var, weight_type, res, model_quality):
        coef_rows = ""
        for i, name in enumerate(res["variable_names"]):
            coeff = res["coefficients"][i]
            se = res["std_errors"][i]
            t_stat = res["t_statistics"][i]
            p_val = res["p_values"][i]
            p_formatted = f"{p_val:.6f}" if p_val >= 0.0001 else "< 0.0001"
            p_class = "significant" if p_val < 0.05 else "non-significant"
            coef_rows += (
                f'<tr class="{p_class}"><td><strong>{html.escape(name)}</strong></td>'
                f"<td>{coeff:.6f}</td><td>{se:.6f}</td><td>{t_stat:.4f}</td><td>{p_formatted}</td></tr>"
            )

        rho = res["rho"]
        rho_note = (
            f"rho = {rho:.4f} ({'significant' if res['rho_p'] < 0.05 else 'not significant'}, p={res['rho_p']:.4f}). "
            + ("A positive, significant rho indicates genuine spatial spillover/feedback beyond what WX alone captures."
               if res['rho_p'] < 0.05 else
               "A non-significant rho suggests the WX terms may already capture most of the spatial structure; consider comparing against a plain OLS-with-WX (Spatial Durbin without the endogenous lag) specification.")
        )

        residual_chart = scatter_plot_svg(
            res["fitted"].tolist(),
            res["residuals"].tolist(),
            x_label="Fitted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )

        guidance_html = analyst_guidance_html(
            "Spatial Durbin Model",
            "SDM lags both the dependent variable and every explanatory variable, nesting Spatial Lag and a restricted Spatial Error form as special cases, estimated here by Spatial 2SLS.",
            [
                "Lagrange Multiplier Diagnostics indicated both robust LM-lag and robust LM-error are significant, motivating a combined specification.",
                "The instrument set (WX, W^2X) has genuine explanatory power for Wy - check this isn't a near-collinear, weak-instrument situation.",
                "Sample size is large enough relative to the doubled parameter count (Intercept + X + WX + rho).",
            ],
            [
                "Interpreting raw beta/theta coefficients as marginal effects directly - SDM requires the LeSage & Pace direct/indirect/total decomposition for that.",
                "A rho estimate outside a plausible range (|rho| close to or above 1) - suggests instrument weakness or model misspecification.",
                "Ignoring model quality warnings below - WX terms are, by construction, correlated with X, raising multicollinearity risk.",
            ],
            [
                "Lagrange Multiplier Diagnostics to re-confirm the specification choice",
                "Spatial Autoregression (SAR) or Spatial Error Regression (SEM) as simpler nested alternatives to compare against",
                "Model Comparison Matrix to compare SDM's fit against the simpler spatial specifications",
            ],
            "SDM is the right tool when Lagrange Multiplier Diagnostics flags both spatial lag AND spatial error dependence, or when theory suggests genuine spillovers in both the outcome and its drivers - treat the coefficient table as a specification screen, not final marginal effects.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Spatial Durbin Model Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 880px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 30px; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 6px; text-align: center; }}
    .card-title {{ font-size: 0.75rem; text-transform: uppercase; color: #4a5568; margin-bottom: 5px; font-weight: 700; letter-spacing: 0.05em; }}
    .card-value {{ font-size: 1.4rem; font-weight: 800; color: #2b6cb0; }}
    h2 {{ font-size: 1.2rem; color: #2d3748; margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #3182ce; padding-left: 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 0.9rem; }}
    th {{ background-color: #ebf8ff; color: #2b6cb0; font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    .significant {{ background-color: #f0fff4; }}
    .non-significant {{ color: #718096; }}
    .note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
    {analyst_guidance_css()}
    {chart_css()}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Spatial Durbin Model (SDM)</h1>
        <p class="subtitle">Dependent Variable: <strong>{html.escape(dep_var)}</strong> | N: <strong>{res["n"]}</strong> | Weights: <strong>{html.escape(weight_type)}</strong> | Estimator: <strong>Spatial 2SLS</strong></p>
    </header>

    <div class="grid">
        <div class="card"><div class="card-title">R-Squared</div><div class="card-value">{res["r2"]:.6f}</div></div>
        <div class="card"><div class="card-title">rho (spatial lag coef.)</div><div class="card-value">{rho:.6f}</div></div>
        <div class="card"><div class="card-title">Residual DF</div><div class="card-value">{res["df_err"]}</div></div>
        <div class="card"><div class="card-title">Parameters</div><div class="card-value">{res["k"]}</div></div>
    </div>

    <h2>Variable Estimates</h2>
    <table>
        <thead><tr><th>Variable Name</th><th>Coefficient</th><th>Std Error</th><th>t-Statistic</th><th>p-value</th></tr></thead>
        <tbody>{coef_rows}</tbody>
    </table>

    <div class="note">{rho_note}</div>

    <h2>Residual vs. Fitted</h2>
    {residual_chart}

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

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying SDM residual styling...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats Spatial Durbin Model residual output",
            {
                "residual": "SDM residual: observed minus fitted dependent value",
                "fitted": "SDM fitted (predicted) dependent value",
            },
            self.displayName(),
        )

        residual_values = [
            f["residual"] for f in layer.getFeatures() if f["residual"] is not None and f["residual"] != NULL
        ]
        std = float(np.std(residual_values)) if residual_values else 1.0
        if std <= 0:
            std = 1.0
        ranges = []
        range_definitions = [
            (-9999.0, -2.5 * std, "#2166ac", "Large Underprediction"),
            (-2.5 * std, -0.5 * std, "#92c5de", "Moderate Underprediction"),
            (-0.5 * std, 0.5 * std, "#f7f7f7", "Near Zero"),
            (0.5 * std, 2.5 * std, "#f4a582", "Moderate Overprediction"),
            (2.5 * std, 9999.0, "#b2182b", "Large Overprediction"),
        ]
        for min_v, max_v, color_hex, label in range_definitions:
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)
            if symbol.symbolLayerCount() > 0:
                sl = symbol.symbolLayer(0)
                if hasattr(sl, "setStrokeColor"):
                    sl.setStrokeColor(QColor("#b0b0b0"))
                if hasattr(sl, "setStrokeWidth"):
                    sl.setStrokeWidth(0.1)
            ranges.append(QgsRendererRange(min_v, max_v, symbol, label))

        renderer = QgsGraduatedSymbolRenderer("residual", ranges)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {}
