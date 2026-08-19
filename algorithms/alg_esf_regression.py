# -*- coding: utf-8 -*-
"""Eigenvector Spatial Filtering (ESF) Regression Processing Algorithm."""
from __future__ import annotations

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
from ..core.advanced_stats_engines import select_esf_eigenvectors
from ..core.analysis_diagnostics import (
    regression_quality_html,
    regression_quality_summary,
    residual_spatial_autocorrelation_html,
    residual_spatial_autocorrelation_summary,
)
from ..core.layer_metadata import apply_output_metadata
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg

from ._icons import algorithm_icon


class ESFRegressionAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    MAX_EIGENVECTORS = "MAX_EIGENVECTORS"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "esf_regression"

    def displayName(self) -> str:
        return "Eigenvector Spatial Filtering (ESF) Regression"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("esf_regression")

    def createInstance(self):
        return ESFRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Griffith's (2003) alternative to Maximum-Likelihood spatial "
            "regression: instead of estimating a spatial autoregressive "
            "parameter (rho or lambda), ESF adds a small set of "
            "'spatial filter' eigenvectors of the (doubly-centered, "
            "symmetrized) spatial weights matrix as extra predictors, then "
            "fits the whole model by plain OLS. Each eigenvector represents "
            "a distinct achievable spatial pattern under the chosen weights, "
            "ranked by the Moran's I it corresponds to; a forward stepwise "
            "procedure adds eigenvectors one at a time - always the one "
            "most correlated with the current residual - stopping once the "
            "residual's Global Moran's I is no longer significant (or a "
            "maximum eigenvector count is reached).\n\n"
            "The appeal: no endogeneity, no log-determinant/Jacobian term, "
            "no numerical optimization over rho or lambda - once the "
            "eigenvectors are selected, it is exactly OLS, so every OLS "
            "diagnostic and standard error formula applies without "
            "modification. The cost: the selected eigenvectors are a "
            "purely data-driven spatial filter with no direct substantive "
            "interpretation of their own (they are not the 'true' spatial "
            "process, just a basis that happens to soak up the residual "
            "autocorrelation) - use them to CONTROL for spatial dependence "
            "so your X coefficients are trustworthy, not to explain WHY "
            "the spatial pattern exists.\n\n"
            "Output fields: residual, fitted (predicted) value, and "
            "spatial_filter (the eigenvector-only contribution to the "
            "fitted value - useful to map on its own, since it isolates "
            "exactly the spatial structure the filter absorbed). The HTML "
            "report shows the selected eigenvector count, their "
            "individual candidate Moran's I values, the final coefficient "
            "table for the ORIGINAL X variables (interpretable as usual "
            "since eigenvectors are orthogonal to X's spatial-lag content "
            "by construction), and the before/after residual Moran's I."
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
            QgsProcessingParameterNumber(
                self.MAX_EIGENVECTORS,
                "Maximum eigenvectors to select",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=1,
                maxValue=50,
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
                "Output ESF HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Eigenvector Spatial Filtering diagnostic report"))

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
        max_eigenvectors = self.parameterAsInt(parameters, self.MAX_EIGENVECTORS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "esf_regression_report.html")

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
            feedback.setProgress(int(15 * (idx / total)))

        n = len(dep_vals)
        p = len(indep_fields)
        if n <= p + 1 + max_eigenvectors:
            raise QgsProcessingException(
                f"Insufficient valid observations ({n}) relative to {p} predictor(s) plus up to {max_eigenvectors} eigenvector(s)."
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

        X_baseline = np.column_stack((np.ones(n), X_data))
        baseline_coefs = np.linalg.pinv(X_baseline.T @ X_baseline) @ X_baseline.T @ y
        baseline_residuals = y - X_baseline @ baseline_coefs
        before_summary = residual_spatial_autocorrelation_summary(baseline_residuals, neighbors, weights, valid_fids)

        feedback.pushInfo(f"Selecting spatial filter eigenvectors (up to {max_eigenvectors})...")
        esf_result = select_esf_eigenvectors(
            y, X_data, neighbors, weights, valid_fids, max_eigenvectors=max_eigenvectors
        )
        n_selected = esf_result["eigenvectors"].shape[1]
        feedback.pushInfo(f"Selected {n_selected} spatial filter eigenvector(s) out of {esf_result['n_candidates']} positive-eigenvalue candidate(s).")

        X_full = np.column_stack((np.ones(n), X_data))
        if n_selected > 0:
            design = np.column_stack((X_full, esf_result["eigenvectors"]))
        else:
            design = X_full
            feedback.pushWarning("No eigenvectors were selected; residual spatial autocorrelation may already be non-significant, or max_eigenvectors is too low.")

        design_inv = np.linalg.pinv(design.T @ design)
        coefs = design_inv @ design.T @ y
        fitted = design @ coefs
        residuals = y - fitted
        df_err = n - design.shape[1]
        if df_err <= 0:
            raise QgsProcessingException("Insufficient degrees of freedom after adding spatial filter eigenvectors.")
        sigma2 = float(residuals @ residuals) / df_err
        cov = sigma2 * design_inv
        se = np.sqrt(np.maximum(0.0, np.diagonal(cov)))

        import math as _math
        n_x_coefs = 1 + p
        t_stats = np.zeros(n_x_coefs)
        p_vals = np.ones(n_x_coefs)
        for j in range(n_x_coefs):
            if se[j] > 0:
                t_stats[j] = coefs[j] / se[j]
                p_vals[j] = 2.0 * (1.0 - 0.5 * (1.0 + _math.erf(abs(t_stats[j]) / _math.sqrt(2.0))))

        ss_res = float(residuals @ residuals)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        if n_selected > 0:
            spatial_filter_contribution = esf_result["eigenvectors"] @ coefs[n_x_coefs:]
        else:
            spatial_filter_contribution = np.zeros(n)

        after_summary = residual_spatial_autocorrelation_summary(residuals, neighbors, weights, valid_fids)
        feedback.pushInfo(
            f"Residual Moran's I: before={before_summary.get('moran_i')}, after={after_summary.get('moran_i')}"
        )

        if feedback.isCanceled():
            return {}

        out_fields = source.fields()
        out_fields.append(QgsField("residual", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("fitted", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("spatial_filter", QVariant.Double, len=12, prec=6))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs()
        )
        self.out_layer_id = dest_id

        results_map = {
            valid_fids[i]: (residuals[i], fitted[i], spatial_filter_contribution[i])
            for i in range(n)
        }

        feedback.pushInfo("Writing residual attributes...")
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)
            fid = f.id()
            if fid in results_map:
                res, fit, sf = results_map[fid]
                out_feat.setAttribute("residual", float(res))
                out_feat.setAttribute("fitted", float(fit))
                out_feat.setAttribute("spatial_filter", float(sf))
            else:
                out_feat.setAttribute("residual", None)
                out_feat.setAttribute("fitted", None)
                out_feat.setAttribute("spatial_filter", None)
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 30 * (current / total)))

        feedback.pushInfo("Generating ESF HTML diagnostics report...")
        self.write_html_report(
            html_path, dep_var, weight_type, indep_fields, coefs, se, t_stats, p_vals,
            n_x_coefs, r2, n, df_err, esf_result, before_summary, after_summary, model_quality,
            fitted, residuals,
        )

        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(
        self, path, dep_var, weight_type, indep_fields, coefs, se, t_stats, p_vals,
        n_x_coefs, r2, n, df_err, esf_result, before_summary, after_summary, model_quality,
        fitted, residuals,
    ):
        variable_names = ["Intercept"] + list(indep_fields)
        coef_rows = ""
        for i, name in enumerate(variable_names):
            p_val = p_vals[i]
            p_formatted = f"{p_val:.6f}" if p_val >= 0.0001 else "< 0.0001"
            p_class = "significant" if p_val < 0.05 else "non-significant"
            coef_rows += (
                f'<tr class="{p_class}"><td><strong>{html.escape(name)}</strong></td>'
                f"<td>{coefs[i]:.6f}</td><td>{se[i]:.6f}</td><td>{t_stats[i]:.4f}</td><td>{p_formatted}</td></tr>"
            )

        eig_rows = "".join(
            f"<tr><td>Eigenvector {idx + 1}</td><td>{val:.4f}</td></tr>"
            for idx, val in enumerate(esf_result["eigenvalues"])
        )
        if not eig_rows:
            eig_rows = '<tr><td colspan="2">No eigenvectors selected.</td></tr>'

        residual_chart = scatter_plot_svg(
            fitted.tolist(),
            residuals.tolist(),
            x_label="Fitted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )
        guidance_html = analyst_guidance_html(
            "Eigenvector Spatial Filtering",
            "ESF adds a small, data-driven set of spatial-pattern eigenvectors as extra OLS predictors, filtering residual spatial autocorrelation without estimating an autoregressive spatial parameter.",
            [
                "Residual Moran's I was significant before filtering and is meaningfully reduced after.",
                "The X-variable coefficients and their significance are stable compared to a plain OLS run without the filter.",
                "The number of selected eigenvectors is small relative to sample size (a large fraction indicates the filter is doing too much work).",
            ],
            [
                "Interpreting individual eigenvectors substantively - they are a mathematical basis, not a measured variable.",
                "Selecting so many eigenvectors that the model effectively memorizes the data (check residual DF and adjusted fit, not just Moran's I).",
                "Comparing ESF's R2 directly against SAR/SEM's R2 without noting they answer different specification questions.",
            ],
            [
                "Lagrange Multiplier Diagnostics as an alternative diagnostic for whether spatial dependence exists at all",
                "OLS Regression to compare the plain-OLS baseline residual Moran's I against this filtered result",
                "Spatial Durbin Model if a fully parametric spatial specification is preferred over a filter-based approach",
            ],
            "Use ESF when the goal is trustworthy X-variable inference in the presence of nuisance spatial autocorrelation - it is a control strategy, not a tool for explaining the spatial process itself.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Eigenvector Spatial Filtering Report</title>
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
    {analyst_guidance_css()}
    {chart_css()}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Eigenvector Spatial Filtering (ESF) Regression</h1>
        <p class="subtitle">Dependent Variable: <strong>{html.escape(dep_var)}</strong> | N: <strong>{n}</strong> | Weights: <strong>{html.escape(weight_type)}</strong></p>
    </header>

    <div class="grid">
        <div class="card"><div class="card-title">R-Squared</div><div class="card-value">{r2:.6f}</div></div>
        <div class="card"><div class="card-title">Eigenvectors Selected</div><div class="card-value">{esf_result["eigenvectors"].shape[1]}</div></div>
        <div class="card"><div class="card-title">Candidates Available</div><div class="card-value">{esf_result["n_candidates"]}</div></div>
        <div class="card"><div class="card-title">Residual DF</div><div class="card-value">{df_err}</div></div>
    </div>

    <h2>Variable Estimates (X variables only)</h2>
    <table>
        <thead><tr><th>Variable Name</th><th>Coefficient</th><th>Std Error</th><th>t-Statistic</th><th>p-value</th></tr></thead>
        <tbody>{coef_rows}</tbody>
    </table>

    <h2>Selected Spatial Filter Eigenvectors</h2>
    <table>
        <thead><tr><th>Selection Order</th><th>Eigenvalue (candidate Moran's I strength)</th></tr></thead>
        <tbody>{eig_rows}</tbody>
    </table>

    <h2>Residual Spatial Autocorrelation: Before Filtering</h2>
    {residual_spatial_autocorrelation_html(before_summary)}

    <h2>Residual Spatial Autocorrelation: After Filtering</h2>
    {residual_spatial_autocorrelation_html(after_summary)}

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

        feedback.pushInfo("Applying ESF residual styling...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats Eigenvector Spatial Filtering residual output",
            {
                "residual": "ESF residual: observed minus fitted dependent value",
                "fitted": "ESF fitted (predicted) dependent value",
                "spatial_filter": "Contribution of the selected spatial filter eigenvectors alone to the fitted value",
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
