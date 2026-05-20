# -*- coding: utf-8 -*-
"""Exploratory Regression Processing Algorithm."""
from __future__ import annotations

import html
import logging
import math
import os
import tempfile

import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)

from ..core.stats_engines import calculate_exploratory_regression
from ..core.analysis_diagnostics import regression_quality_html, regression_quality_summary

logger = logging.getLogger("PlanX GeoStats Lab")


class ExploratoryRegressionAlgorithm(QgsProcessingAlgorithm):
    MAX_MODEL_COMBINATIONS = 5000

    INPUT = "INPUT"
    DEPENDENT_FIELD = "DEPENDENT_FIELD"
    EXPLANATORY_FIELDS = "EXPLANATORY_FIELDS"
    MAX_VARIABLES = "MAX_VARIABLES"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "exploratory_regression"

    def displayName(self) -> str:
        return "Exploratory Regression"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def createInstance(self):
        return ExploratoryRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Tests combinations of candidate explanatory variables with ordinary "
            "least squares and ranks the resulting models by corrected Akaike "
            "Information Criterion (AICc).\n\n"
            "Use this tool before committing to a single regression specification. "
            "It is designed as a planning-analysis screening step: it helps compare "
            "which variable sets explain the dependent field most efficiently, while "
            "still leaving final interpretation, diagnostics, and domain judgment to "
            "the analyst. The report lists the best-ranked models, their adjusted R2, "
            "AICc, and fitted coefficients."
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
                self.DEPENDENT_FIELD,
                "Dependent field (numeric)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.EXPLANATORY_FIELDS,
                "Candidate explanatory fields (numeric)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_VARIABLES,
                "Maximum variables per model",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=1,
                maxValue=8,
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
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Exploratory regression report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        dependent_field = self.parameterAsString(parameters, self.DEPENDENT_FIELD, context)
        explanatory_fields = self.parameterAsFields(parameters, self.EXPLANATORY_FIELDS, context)
        max_variables = self.parameterAsInt(parameters, self.MAX_VARIABLES, context)

        if not explanatory_fields:
            raise QgsProcessingException("Select at least one candidate explanatory field.")
        if dependent_field in explanatory_fields:
            raise QgsProcessingException("The dependent field cannot also be an explanatory field.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_exploratory_regression.html")

        fields = source.fields()
        dep_idx = fields.lookupField(dependent_field)
        exp_indices = [fields.lookupField(name) for name in explanatory_fields]
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dependent_field}' not found.")
        missing = [name for name, idx in zip(explanatory_fields, exp_indices) if idx < 0]
        if missing:
            raise QgsProcessingException(f"Explanatory fields not found: {', '.join(missing)}")

        y_values = []
        x_rows = []
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
            feedback.setProgress(int(35 * (idx / total)))

        n_records = len(y_values)
        if n_records < 5:
            raise QgsProcessingException(
                f"At least 5 complete numeric records are required; found {n_records}."
            )

        max_allowed = max(1, min(max_variables, len(explanatory_fields), n_records - 2))
        combination_count = self._count_model_combinations(len(explanatory_fields), max_allowed)
        if combination_count > self.MAX_MODEL_COMBINATIONS:
            raise QgsProcessingException(
                "This exploratory regression setup would estimate "
                f"{combination_count:,} candidate models, which is above the safety limit of "
                f"{self.MAX_MODEL_COMBINATIONS:,}. Reduce the number of candidate fields or "
                "lower the maximum variables per model."
            )
        feedback.pushInfo(
            f"Testing {combination_count} OLS combinations with up to "
            f"{max_allowed} explanatory variables..."
        )
        y_array = np.array(y_values, dtype=float)
        x_array = np.array(x_rows, dtype=float)
        model_quality = regression_quality_summary(y_array, x_array, explanatory_fields, source.featureCount())
        for risk in model_quality["risks"]:
            feedback.pushWarning(risk)

        models = calculate_exploratory_regression(
            y_array,
            x_array,
            explanatory_fields,
            max_allowed,
        )
        if not models:
            raise QgsProcessingException("No valid regression models could be estimated.")

        self._write_html(
            html_path,
            dependent_field,
            explanatory_fields,
            n_records,
            skipped,
            max_allowed,
            combination_count,
            models,
            model_quality,
        )
        feedback.setProgress(100)
        feedback.pushInfo(f"Best model AICc: {models[0]['aicc']:.4f}")
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

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

    def _write_html(
        self,
        path,
        dependent_field,
        explanatory_fields,
        n_records,
        skipped,
        max_allowed,
        combination_count,
        models,
        model_quality,
    ):
        top_models = models[:20]
        rows = []
        best_aicc = models[0]["aicc"]
        for rank, model in enumerate(top_models, start=1):
            variables = ", ".join(html.escape(name) for name in model["variables"])
            delta_aicc = model["aicc"] - best_aicc
            rank_note = "Best AICc" if rank == 1 else f"Delta AICc {delta_aicc:.3f}"
            coefficient_lines = "<br>".join(
                f"{html.escape(name)}: {value:.6g}"
                for name, value in model["coefficients"].items()
            )
            rows.append(
                "<tr>"
                f"<td>{rank}</td>"
                f"<td>{variables}</td>"
                f"<td>{model['n_vars']}</td>"
                f"<td>{model['r2']:.4f}</td>"
                f"<td>{model['adj_r2']:.4f}</td>"
                f"<td>{model['aicc']:.4f}</td>"
                f"<td>{rank_note}</td>"
                f"<td>{coefficient_lines}</td>"
                "</tr>"
            )

        best = models[0]
        candidate_text = ", ".join(html.escape(name) for name in explanatory_fields)
        best_vars = ", ".join(html.escape(name) for name in best["variables"])
        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Exploratory Regression</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #25313f; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 1120px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.7rem; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 16px 18px; margin: 20px 0; }}
.note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .86rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6) {{ font-family: Consolas, monospace; }}
</style>
</head>
<body>
<div class="container">
<h1>Exploratory Regression</h1>
<p class="subtitle">Dependent field: <strong>{html.escape(dependent_field)}</strong></p>
<div class="summary">
<strong>Best ranked model:</strong> {best_vars}<br>
Adjusted R2: <strong>{best['adj_r2']:.4f}</strong> | AICc: <strong>{best['aicc']:.4f}</strong>
</div>
<p><strong>Complete records used:</strong> {n_records}<br>
<strong>Skipped records with missing/non-numeric values:</strong> {skipped}<br>
<strong>Candidate explanatory fields:</strong> {candidate_text}<br>
<strong>Maximum variables per tested model:</strong> {max_allowed}<br>
<strong>Candidate models estimated:</strong> {combination_count}</p>
<div class="note">
Exploratory regression is a screening tool, not an automatic model approval step. Models are ranked by AICc, which rewards fit while penalizing unnecessary complexity. Review coefficient signs, field meaning, residual behavior, spatial autocorrelation, and planning theory before selecting a final model.
</div>
{regression_quality_html(model_quality)}
<div class="note">
<strong>Recommended next action:</strong> rerun the best candidate specification in Ordinary Least Squares (OLS) Regression and inspect residual diagnostics before using the model for scenario evaluation or policy interpretation.
</div>
<table>
<thead>
<tr><th>Rank</th><th>Variables</th><th>Count</th><th>R2</th><th>Adjusted R2</th><th>AICc</th><th>Rank Reason</th><th>Coefficients</th></tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _count_model_combinations(self, n_fields, max_variables):
        total = 0
        upper = min(n_fields, max_variables)
        for size in range(1, upper + 1):
            total += math.comb(n_fields, size)
        return total
