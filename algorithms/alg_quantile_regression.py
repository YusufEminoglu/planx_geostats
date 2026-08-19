# -*- coding: utf-8 -*-
"""Quantile Regression Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg
from ..core.stats_engines import calculate_quantile_regression

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class QuantileRegressionAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    TAU = "TAU"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "quantile_regression"

    def displayName(self) -> str:
        return "Quantile Regression"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("quantile_regression")

    def createInstance(self):
        return QuantileRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a linear model for one conditional quantile (Tau) of the "
            "dependent variable, rather than its conditional mean (what OLS/"
            "Generalized Linear Regression estimate). At Tau=0.5 this is "
            "median regression, more resistant to outliers than mean-based "
            "OLS; at Tau=0.9 it describes what drives the upper tail of the "
            "outcome, which can have a genuinely different relationship with "
            "the explanatory fields than the average case.\n\n"
            "This matters for skewed planning outcomes (housing prices, "
            "traffic counts, permit-processing times) where the factors "
            "driving typical cases and extreme cases often differ - a field "
            "with little effect on the median can be the dominant driver of "
            "the top 10%. Run this tool at several Tau values (e.g. 0.1, "
            "0.5, 0.9) on the same fields and compare coefficients across "
            "runs to see whether a relationship is constant across the "
            "distribution or concentrated at one end.\n\n"
            "Fitted by iteratively reweighted least squares on the pinball "
            "loss (no external dependency), which converges to the standard "
            "linear-programming quantile-regression solution for "
            "well-behaved data. Output: fitted values and residuals per "
            "complete record, plus an HTML report with coefficients and a "
            "pseudo-R2 (Koenker & Machado R1) comparing this model's fit "
            "against an intercept-only quantile.\n\n"
            "No standard errors or p-values are reported - quantile-"
            "regression inference needs bootstrap or sandwich-estimator "
            "methods beyond this tool's scope; read coefficients "
            "descriptively (direction and relative magnitude across Tau "
            "values) rather than testing them for significance."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEP_VAR, "Dependent variable field", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.INDEPENDENTS, "Explanatory variable fields", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.TAU, "Quantile (Tau)", type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5, minValue=0.01, maxValue=0.99,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output quantile regression predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output quantile regression HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Quantile regression report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        indep_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)
        tau = self.parameterAsDouble(parameters, self.TAU, context)
        if not indep_fields:
            raise QgsProcessingException("At least one explanatory variable must be selected.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_quantile_regression_report.html")

        fields = source.fields()
        dep_idx = fields.lookupField(dep_var)
        exp_indices = [fields.lookupField(name) for name in indep_fields]
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        missing = [name for name, idx in zip(indep_fields, exp_indices) if idx < 0]
        if missing:
            raise QgsProcessingException(f"Explanatory fields not found: {', '.join(missing)}")

        y_values, x_rows, valid_fids = [], [], []
        skipped = 0
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting complete numeric records...")
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            dep_value = self._to_float(feature.attribute(dep_idx))
            exp_values = [self._to_float(feature.attribute(field_idx)) for field_idx in exp_indices]
            if dep_value is None or any(value is None for value in exp_values):
                skipped += 1
                continue
            y_values.append(dep_value)
            x_rows.append(exp_values)
            valid_fids.append(feature.id())
            feedback.setProgress(int(30 * (idx / total)))

        if len(y_values) <= len(indep_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y_values)}) for {len(indep_fields)} explanatory variable(s)."
            )

        y = np.array(y_values, dtype=float)
        x_data = np.array(x_rows, dtype=float)
        feedback.pushInfo(f"Fitting quantile regression at Tau={tau}...")
        try:
            results = calculate_quantile_regression(y, x_data, tau=tau)
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        pseudo_r2_text = f"{results['pseudo_r2']:.4f}" if results["pseudo_r2"] is not None else "n/a"
        feedback.pushInfo(f"Converged={results['converged']}; iterations={results['iterations']}; pseudo-R2={pseudo_r2_text}")

        out_fields = source.fields()
        out_fields.append(QgsField("qr_fit", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("qr_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("qr_used", QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        result_map = {fid: idx for idx, fid in enumerate(valid_fids)}
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                row_idx = result_map[fid]
                out_feature.setAttribute("qr_fit", float(results["fitted"][row_idx]))
                out_feature.setAttribute("qr_resid", float(results["residuals"][row_idx]))
                out_feature.setAttribute("qr_used", 1)
            else:
                out_feature.setAttribute("qr_fit", None)
                out_feature.setAttribute("qr_resid", None)
                out_feature.setAttribute("qr_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, dep_var, indep_fields, tau, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats quantile regression output",
            {
                "qr_fit": "Fitted Tau-quantile value",
                "qr_resid": "Observed minus fitted",
                "qr_used": "1 if the record had complete values and was used to fit and predict, 0 otherwise",
            },
            "quantile_regression",
        )
        return {}

    def _to_float(self, value):
        if value is None or value == NULL or str(value) == "NULL":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _write_html(self, path, dep_var, indep_fields, tau, results, skipped):
        names = ["Intercept"] + list(indep_fields)
        coef_rows = "".join(
            f"<tr><td><strong>{html.escape(name)}</strong></td><td>{coef:.6f}</td></tr>"
            for name, coef in zip(names, results["coefficients"])
        )
        pseudo_r2_text = f"{results['pseudo_r2']:.6f}" if results["pseudo_r2"] is not None else "n/a"
        residual_chart = scatter_plot_svg(
            results["fitted"].tolist() if hasattr(results["fitted"], "tolist") else list(results["fitted"]),
            results["residuals"].tolist() if hasattr(results["residuals"], "tolist") else list(results["residuals"]),
            x_label="Fitted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )
        guidance = analyst_guidance_html(
            "Quantile Regression",
            f"A linear model for the Tau={tau:g} conditional quantile, "
            "fit by iteratively reweighted least squares on the pinball loss.",
            [
                "The model converged before the iteration cap.",
                "Coefficients are compared across several Tau values (0.1/0.5/0.9), not read from a single run alone.",
            ],
            [
                "Did not converge (try more complete records or fewer explanatory fields).",
                "Pseudo-R2 well below the OLS/GLR R2 on the same fields at the median (Tau=0.5) - the linear specification may fit the mean much better than the median for this outcome.",
            ],
            [
                "Rerun at other Tau values (0.1, 0.25, 0.75, 0.9) and compare coefficient patterns.",
                "Generalized Linear Regression - the conditional-mean baseline to contrast against.",
            ],
            "Coefficients here describe association with the chosen "
            "quantile, not the mean - do not mix them with OLS coefficients "
            "in the same sentence without noting which quantile they refer to.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Quantile Regression Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .9rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>Quantile Regression (Tau={tau:g})</h1>
<p>Dependent: <strong>{html.escape(dep_var)}</strong> | Explanatory fields: <strong>{html.escape(', '.join(indep_fields))}</strong></p>
<div class="summary">
Converged = {results['converged']} ({results['iterations']} iterations) | Pseudo-R2 = {pseudo_r2_text}<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
<h2>Coefficients</h2>
<table><thead><tr><th>Field</th><th>Coefficient</th></tr></thead><tbody>{coef_rows}</tbody></table>
<h2>Residual vs. Fitted</h2>
{residual_chart}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
