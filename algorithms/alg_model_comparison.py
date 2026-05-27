# -*- coding: utf-8 -*-
"""Model Comparison Matrix Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from qgis.core import (
    NULL,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterMultipleLayers,
    QgsWkbTypes,
)

from ..core.analysis_diagnostics import (
    filter_weights_to_valid_ids,
    format_number,
    model_fit_summary,
    residual_spatial_autocorrelation_summary,
)
from ..core.model_audit import assign_model_scores, model_recommendation
from ..core.weights import build_weights_matrix

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class ModelComparisonAlgorithm(QgsProcessingAlgorithm):
    MODEL_LAYERS = "MODEL_LAYERS"
    DEP_VAR = "DEP_VAR"
    HTML_REPORT = "HTML_REPORT"

    MODEL_SPECS = [
        {
            "name": "OLS Regression",
            "required": ["residual"],
            "residual": "residual",
            "predicted": None,
            "used": None,
            "stdres": "std_res",
        },
        {
            "name": "Generalized Linear Regression",
            "required": ["glr_fit", "glr_resid"],
            "residual": "glr_resid",
            "predicted": "glr_fit",
            "used": "glr_used",
            "stdres": None,
        },
        {
            "name": "GWR",
            "required": ["y_predicted", "residual"],
            "residual": "residual",
            "predicted": "y_predicted",
            "used": None,
            "stdres": None,
        },
        {
            "name": "MGWR",
            "required": ["mgwr_pred", "mgwr_resid"],
            "residual": "mgwr_resid",
            "predicted": "mgwr_pred",
            "used": "mgwr_used",
            "stdres": "mgwr_std",
        },
        {
            "name": "Spatial Lag Regression",
            "required": ["sar_pred", "sar_resid"],
            "residual": "sar_resid",
            "predicted": "sar_pred",
            "used": "sar_used",
            "stdres": "sar_stdres",
        },
        {
            "name": "Spatial Error Regression",
            "required": ["sem_pred", "sem_resid"],
            "residual": "sem_resid",
            "predicted": "sem_pred",
            "used": "sem_used",
            "stdres": "sem_stdres",
        },
    ]

    def name(self) -> str:
        return "model_comparison_matrix"

    def displayName(self) -> str:
        return "Model Comparison Matrix"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("model_comparison_matrix")

    def createInstance(self):
        return ModelComparisonAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Compares multiple PlanX GeoStats model output layers in a single audit report. "
            "The tool recognizes OLS, GLR, GWR, MGWR, Spatial Lag, and Spatial Error outputs "
            "from their standard diagnostic fields, then calculates comparable R2, RMSE, MAE, "
            "bias, complete-record coverage, and residual spatial autocorrelation.\n\n"
            "Use this after running several candidate models with the same dependent variable. "
            "The report is designed for analyst review: lower error is not the only criterion; "
            "residual spatial structure, model assumptions, variable meaning, and planning theory "
            "must be considered together."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.MODEL_LAYERS,
                "PlanX model output layers to compare",
                layerType=QgsProcessing.TypeVectorAnyGeometry,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEP_VAR,
                "Observed dependent variable field",
                parentLayerParameterName=self.MODEL_LAYERS,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output model comparison HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Model comparison report"))

    def processAlgorithm(self, parameters, context, feedback):
        layers = self.parameterAsLayerList(parameters, self.MODEL_LAYERS, context)
        if not layers:
            raise QgsProcessingException("Select at least one PlanX model output layer.")
        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_model_comparison_matrix.html")

        comparisons = []
        for layer in layers:
            if feedback.isCanceled():
                break
            comparison = self._summarize_layer(layer, dep_var, feedback)
            comparisons.append(comparison)
            if comparison.get("usable"):
                feedback.pushInfo(
                    f"{comparison['layer_name']}: {comparison['model_name']} "
                    f"R2={format_number(comparison['fit']['r2'], 6)}, "
                    f"RMSE={format_number(comparison['fit']['rmse'], 6)}."
                )
            else:
                feedback.pushWarning(f"{comparison['layer_name']}: {comparison['message']}")

        usable = [item for item in comparisons if item.get("usable")]
        if not usable:
            raise QgsProcessingException("None of the selected layers contained recognized PlanX model output fields.")

        self._write_html(html_path, dep_var, comparisons)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _summarize_layer(self, layer, dep_var, feedback):
        fields = layer.fields()
        field_names = [field.name() for field in fields]
        dep_idx = fields.lookupField(dep_var)
        if dep_idx < 0:
            return self._unusable(layer, f"Dependent field '{dep_var}' was not found.")

        spec = self._detect_model(field_names)
        if spec is None:
            return self._unusable(layer, "Recognized PlanX model diagnostic fields were not found.")

        residual_idx = fields.lookupField(spec["residual"])
        predicted_idx = fields.lookupField(spec["predicted"]) if spec["predicted"] else -1
        used_idx = fields.lookupField(spec["used"]) if spec["used"] else -1

        observed = []
        predicted = []
        residuals = []
        valid_fids = []
        skipped = 0
        for feature in layer.getFeatures():
            if feedback.isCanceled():
                break
            if used_idx >= 0 and self._to_float(feature.attribute(used_idx)) == 0.0:
                skipped += 1
                continue
            y = self._to_float(feature.attribute(dep_idx))
            residual = self._to_float(feature.attribute(residual_idx))
            if y is None or residual is None:
                skipped += 1
                continue
            pred = self._to_float(feature.attribute(predicted_idx)) if predicted_idx >= 0 else None
            if pred is None:
                pred = y - residual
            observed.append(y)
            predicted.append(pred)
            residuals.append(residual)
            valid_fids.append(int(feature.id()))

        if len(observed) < 2:
            return self._unusable(layer, "Fewer than 2 complete model records were available.")

        try:
            fit = model_fit_summary(observed, predicted, residuals)
        except ValueError as exc:
            return self._unusable(layer, str(exc))

        residual_spatial = self._residual_spatial(layer, valid_fids, residuals, feedback)
        return {
            "usable": True,
            "layer_name": layer.name(),
            "model_name": spec["name"],
            "fit": fit,
            "residual_spatial": residual_spatial,
            "skipped": skipped,
            "total": int(layer.featureCount()),
            "coverage": len(observed) / max(1, int(layer.featureCount())),
        }

    def _detect_model(self, field_names):
        field_set = set(field_names)
        for spec in self.MODEL_SPECS:
            if all(name in field_set for name in spec["required"]):
                return spec
        return None

    def _residual_spatial(self, layer, valid_fids, residuals, feedback):
        try:
            weight_type = "queen" if layer.geometryType() == QgsWkbTypes.PolygonGeometry else "knn"
            neighbors, _, _, _ = build_weights_matrix(layer, weight_type, k_neighbors=8, feedback=feedback)
            filtered_neighbors, filtered_weights, filtered_ids = filter_weights_to_valid_ids(neighbors, valid_fids)
            return residual_spatial_autocorrelation_summary(residuals, filtered_neighbors, filtered_weights, filtered_ids)
        except Exception as exc:
            return {
                "available": False,
                "moran_i": None,
                "expected_i": None,
                "variance": None,
                "z_score": None,
                "p_value": None,
                "neighbor_summary": None,
                "status": "Not available",
                "message": str(exc),
            }

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

    def _unusable(self, layer, message):
        return {
            "usable": False,
            "layer_name": layer.name(),
            "message": message,
        }

    def _write_html(self, path, dep_var, comparisons):
        usable = [item for item in comparisons if item.get("usable")]
        assign_model_scores(usable)
        recommendation = model_recommendation(comparisons)

        rows = []
        for item in sorted(comparisons, key=lambda row: row.get("score", 10**9)):
            if not item.get("usable"):
                rows.append(
                    "<tr class=\"unusable\">"
                    f"<td>{html.escape(item['layer_name'])}</td>"
                    "<td>Not recognized</td>"
                    f"<td colspan=\"11\">{html.escape(item['message'])}</td>"
                    "</tr>"
                )
                continue
            fit = item["fit"]
            residual = item["residual_spatial"]
            p_value = residual.get("p_value")
            residual_status = residual.get("status", "n/a")
            row_class = "warning" if p_value is not None and p_value < 0.05 else ""
            rows.append(
                f"<tr class=\"{row_class}\">"
                f"<td>{html.escape(item['layer_name'])}</td>"
                f"<td>{html.escape(item['model_name'])}</td>"
                f"<td>{item['rank']}</td>"
                f"<td>{format_number(item['score'], 3)}</td>"
                f"<td>{fit['n']}</td>"
                f"<td>{item['coverage']:.1%}</td>"
                f"<td>{format_number(fit['r2'], 6)}</td>"
                f"<td>{format_number(fit['rmse'], 6)}</td>"
                f"<td>{format_number(fit['mae'], 6)}</td>"
                f"<td>{format_number(fit['bias'], 6)}</td>"
                f"<td>{format_number(residual.get('moran_i'), 6)}</td>"
                f"<td>{format_number(residual.get('p_value'), 6)}</td>"
                f"<td>{html.escape(residual_status)}</td>"
                "</tr>"
            )

        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Model Comparison Matrix</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #25313f; background: #f6f8fb; margin: 0; padding: 24px; line-height: 1.55; }}
.container {{ max-width: 1180px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.72rem; color: #17212f; }}
h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; border-left: 4px solid #2b6cb0; padding-left: 10px; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 16px 18px; margin: 20px 0; }}
.note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 9px; text-align: left; vertical-align: top; font-size: .84rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .70rem; letter-spacing: .05em; }}
.warning {{ background: #fff8e6; }}
.unusable {{ color: #7a4a1f; background: #fffaf0; }}
footer {{ margin-top: 36px; padding-top: 14px; border-top: 1px solid #edf2f7; color: #7a899c; font-size: .82rem; }}
</style>
</head>
<body>
<div class="container">
<h1>Model Comparison Matrix</h1>
<p class="subtitle">Observed dependent field: <strong>{html.escape(dep_var)}</strong> | Compared layers: <strong>{len(comparisons)}</strong></p>
<section class="summary"><strong>Executive summary.</strong> {html.escape(recommendation)}</section>

<h2>Comparison Table</h2>
<table>
<thead>
<tr>
<th>Layer</th><th>Model</th><th>Rank</th><th>Score</th><th>N</th><th>Coverage</th><th>R2</th><th>RMSE</th><th>MAE</th><th>Bias</th><th>Residual Moran's I</th><th>Residual p</th><th>Residual Status</th>
</tr>
</thead>
<tbody>{''.join(rows)}</tbody>
</table>

<h2>How to Read This Report</h2>
<div class="note">
Lower RMSE and MAE indicate better predictive fit on the records available in each output layer, but fit alone is not enough for planning decisions. A model with low error and spatially patterned residuals may still be missing a neighborhood process, boundary effect, or key explanatory variable. Prefer models that are interpretable, defensible for the planning question, and do not leave strong residual spatial structure.
The score is a normalized audit score where lower is better. It combines RMSE rank, MAE rank, residual spatial-pattern penalty, incomplete-record penalty, and missing-diagnostic penalty.
</div>

<h2>Recommended Analyst Action</h2>
<ul>
<li>Open the residual layer for each candidate and inspect where the largest residuals cluster.</li>
<li>Check whether models were run on the same complete records; lower coverage can make metrics incomparable.</li>
<li>Use OLS and GLR as transparent baselines, SAR/SEM for global spatial dependence, and GWR/MGWR for spatially varying relationships.</li>
<li>Document the final choice with model purpose, variable theory, residual behavior, and spatial scale.</li>
</ul>

<footer>Generated by PlanX GeoStats Lab model audit engine.</footer>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
