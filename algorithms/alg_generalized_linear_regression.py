# -*- coding: utf-8 -*-
"""Generalized Linear Regression Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
)

from ..core.analysis_diagnostics import regression_quality_html, regression_quality_summary
from ..core.stats_engines import calculate_glr

logger = logging.getLogger("PlanX GeoStats Lab")


class GeneralizedLinearRegressionAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    FAMILY = "FAMILY"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    FAMILIES = ["Gaussian (continuous / OLS)", "Logistic (binary)", "Poisson (count)"]
    FAMILY_KEYS = ["gaussian", "logistic", "poisson"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "generalized_linear_regression"

    def displayName(self) -> str:
        return "Generalized Linear Regression (GLR)"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def createInstance(self):
        return GeneralizedLinearRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a generalized linear regression model for continuous, binary, or count "
            "dependent variables. Gaussian is equivalent to standard OLS-style linear "
            "regression, Logistic models a binary 0/1 outcome, and Poisson models "
            "non-negative integer counts.\n\n"
            "The output layer includes fitted values and residuals for each complete record. "
            "The HTML report includes coefficients, standard errors, z-statistics, p-values, "
            "AIC, convergence status, and model-quality warnings."
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
                "Explanatory variable fields",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.FAMILY,
                "Model family",
                options=self.FAMILIES,
                defaultValue=0,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output GLR predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output GLR HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "GLR diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        indep_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)
        family_idx = self.parameterAsEnum(parameters, self.FAMILY, context)
        family = self.FAMILY_KEYS[family_idx]
        if not indep_fields:
            raise QgsProcessingException("At least one explanatory variable must be selected.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_glr_report.html")

        fields = source.fields()
        dep_idx = fields.lookupField(dep_var)
        exp_indices = [fields.lookupField(name) for name in indep_fields]
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        missing = [name for name, idx in zip(indep_fields, exp_indices) if idx < 0]
        if missing:
            raise QgsProcessingException(f"Explanatory fields not found: {', '.join(missing)}")

        y_values = []
        x_rows = []
        valid_fids = []
        skipped = 0
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting complete numeric GLR records...")
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
        quality = regression_quality_summary(y, x_data, indep_fields, source.featureCount())
        for risk in quality["risks"]:
            feedback.pushWarning(risk)

        try:
            results = calculate_glr(y, x_data, family)
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        feedback.pushInfo(
            f"GLR fitted using {family} family; converged={results['converged']}; "
            f"iterations={results['iterations']}; AIC={results['aic']:.4f}."
        )

        out_fields = source.fields()
        out_fields.append(QgsField("glr_fit", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("glr_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("glr_used", QVariant.Int))
        if family == "logistic":
            out_fields.append(QgsField("glr_class", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs(),
        )
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
                fitted = float(results["fitted"][row_idx])
                out_feature.setAttribute("glr_fit", fitted)
                out_feature.setAttribute("glr_resid", float(results["residuals"][row_idx]))
                out_feature.setAttribute("glr_used", 1)
                if family == "logistic":
                    out_feature.setAttribute("glr_class", 1 if fitted >= 0.5 else 0)
            else:
                out_feature.setAttribute("glr_fit", None)
                out_feature.setAttribute("glr_resid", None)
                out_feature.setAttribute("glr_used", 0)
                if family == "logistic":
                    out_feature.setAttribute("glr_class", None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, dep_var, indep_fields, family, results, quality, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _to_float(self, value):
        if value is None or value == QVariant() or str(value) == "NULL":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _write_html(self, path, dep_var, indep_fields, family, results, quality, skipped):
        coef_rows = []
        names = ["Intercept"] + list(indep_fields)
        for name, coef, se, z_value, p_value in zip(
            names,
            results["coefficients"],
            results["std_errors"],
            results["z_statistics"],
            results["p_values"],
        ):
            coef_rows.append(
                "<tr>"
                f"<td><strong>{html.escape(name)}</strong></td>"
                f"<td>{coef:.6f}</td>"
                f"<td>{se:.6f}</td>"
                f"<td>{z_value:.4f}</td>"
                f"<td>{p_value:.6f}</td>"
                "</tr>"
            )
        r2_block = f"<p><strong>Gaussian R2:</strong> {results['r2']:.6f}</p>" if results["r2"] is not None else ""
        family_label = {
            "gaussian": "Gaussian continuous model",
            "logistic": "Logistic binary model",
            "poisson": "Poisson count model",
        }[family]
        next_action = (
            "Inspect residuals and model-quality warnings. For logistic and Poisson models, validate predictions against observed classes or counts before scenario use."
        )
        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Generalized Linear Regression</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #25313f; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 1080px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.7rem; }}
h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 16px 18px; margin: 20px 0; }}
.note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
.next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .86rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
</style>
</head>
<body>
<div class="container">
<h1>Generalized Linear Regression (GLR)</h1>
<p class="subtitle">Dependent field: <strong>{html.escape(dep_var)}</strong> | Family: <strong>{family_label}</strong> | Complete records: <strong>{quality['used_records']}</strong> | Skipped: <strong>{skipped}</strong></p>
<div class="summary">
<strong>Converged:</strong> {results['converged']}<br>
<strong>Iterations:</strong> {results['iterations']}<br>
<strong>AIC:</strong> {results['aic']:.4f}<br>
<strong>Log likelihood:</strong> {results['log_likelihood']:.4f}
</div>
{r2_block}
{regression_quality_html(quality)}
<h2>Coefficient Estimates</h2>
<table>
<thead><tr><th>Variable</th><th>Coefficient</th><th>Std Error</th><th>z-statistic</th><th>p-value</th></tr></thead>
<tbody>{''.join(coef_rows)}</tbody>
</table>
<h2>Recommended Analyst Action</h2>
<div class="next-action">{html.escape(next_action)}</div>
<h2>Assumptions and Caveats</h2>
<ul>
<li>Gaussian models continuous outcomes, Logistic requires 0/1 binary outcomes, and Poisson requires non-negative integer counts.</li>
<li>Coefficient significance uses normal approximation; review sample size, multicollinearity, and residual behavior.</li>
<li>GLR is global and does not account for spatial dependence by itself; use residual diagnostics or spatial regression when residuals remain spatially structured.</li>
</ul>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
