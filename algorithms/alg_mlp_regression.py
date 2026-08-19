# -*- coding: utf-8 -*-
"""Neural Network (MLP) Regression Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

from ._mixins import HelpUrlMixin
from qgis.core import (
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
    QgsProcessingParameterString,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.ml_engines import extract_regression_matrix, fit_mlp_regressor
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class MLPRegressionAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    HIDDEN_LAYERS = "HIDDEN_LAYERS"
    MAX_ITER = "MAX_ITER"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "mlp_regression"

    def displayName(self) -> str:
        return "Neural Network Regression (MLP)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("mlp_regression")

    def createInstance(self):
        return MLPRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Multi-Layer Perceptron (a small feed-forward neural network) "
            "regressor: explanatory fields pass through one or more hidden "
            "layers of weighted, non-linearly activated units before producing a "
            "prediction, with weights learned by gradient descent. Features and "
            "target are standardized internally - no manual scaling needed.\n\n"
            "Output: fitted values and residuals per complete record, plus an "
            "HTML report with R2, RMSE, MAE, and the training convergence status "
            "(iterations used versus the maximum allowed). A model that hits the "
            "iteration cap without converging should not be trusted - raise "
            "Maximum iterations and refit, or simplify the hidden-layer sizes.\n\n"
            "MLPs need substantially more complete records than tree-based "
            "models to fit reliably (a rough rule of thumb: at least tens of "
            "records per weight in the network) and provide no feature "
            "importances - on typical single-district or single-city planning "
            "datasets, Random Forest Regression or Gradient Boosting Regression "
            "usually give equal or better accuracy with easier interpretation. "
            "Reach for this tool specifically when you have a large sample and "
            "suspect smooth, highly non-linear interactions that tree splits "
            "approximate poorly.\n\n"
            "Hidden layer sizes are given as comma-separated integers (e.g. "
            "'64,32' for two layers of 64 then 32 units); fewer, smaller layers "
            "reduce overfitting risk on modest planning datasets."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TARGET, "Target field (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FEATURES, "Explanatory fields (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(QgsProcessingParameterString(self.HIDDEN_LAYERS, "Hidden layer sizes (comma-separated)", defaultValue="64,32"))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITER, "Maximum training iterations", type=QgsProcessingParameterNumber.Integer,
                defaultValue=1000, minValue=50, maxValue=20000,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output MLP predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output MLP HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "MLP diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        hidden_text = self.parameterAsString(parameters, self.HIDDEN_LAYERS, context)
        try:
            hidden_layers = tuple(int(part.strip()) for part in hidden_text.split(",") if part.strip())
        except ValueError:
            raise QgsProcessingException(f"Could not parse hidden layer sizes '{hidden_text}' as comma-separated integers.")
        if not hidden_layers:
            raise QgsProcessingException("At least one hidden layer size must be given.")

        max_iter = self.parameterAsInt(parameters, self.MAX_ITER, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_mlp_regression_report.html")

        feedback.pushInfo("Extracting complete numeric records...")
        try:
            extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 30))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, valid_fids, skipped = extraction["x"], extraction["y"], extraction["valid_fids"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric values.")

        feedback.pushInfo(f"Training MLP with hidden layers {hidden_layers}...")
        try:
            results = fit_mlp_regressor(x, y, hidden_layer_sizes=hidden_layers, max_iter=max_iter)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Neural Network Regression (MLP)", ["scikit-learn"], exc))

        feedback.pushInfo(
            f"R2={results['r2']:.4f}, RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}, "
            f"converged={results['converged']} ({results['n_iter']}/{max_iter} iterations)"
        )
        if not results["converged"]:
            feedback.pushWarning("Training hit the maximum iteration cap without converging; results may be unreliable.")

        out_fields = source.fields()
        out_fields.append(QgsField("mlp_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("mlp_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("mlp_used", QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        result_map = {fid: idx for idx, fid in enumerate(valid_fids)}
        total = source.featureCount() or 1
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                row_idx = result_map[fid]
                out_feature.setAttribute("mlp_pred", float(results["fitted"][row_idx]))
                out_feature.setAttribute("mlp_resid", float(results["residuals"][row_idx]))
                out_feature.setAttribute("mlp_used", 1)
            else:
                out_feature.setAttribute("mlp_pred", None)
                out_feature.setAttribute("mlp_resid", None)
                out_feature.setAttribute("mlp_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, hidden_layers, max_iter, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats Neural Network (MLP) regression output",
            {
                "mlp_pred": "MLP predicted value (converted back from the internally standardized fit)",
                "mlp_resid": "Observed minus predicted (positive = model underpredicted)",
                "mlp_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "mlp_regression",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, hidden_layers, max_iter, results, skipped):
        convergence_note = (
            "Converged before the iteration cap." if results["converged"]
            else "<strong>Did not converge</strong> - hit the maximum iteration cap; treat results with caution."
        )
        residual_chart = scatter_plot_svg(
            list(results["fitted"]),
            list(results["residuals"]),
            x_label="Fitted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )
        guidance = analyst_guidance_html(
            "Neural Network Regression (MLP)",
            "A small feed-forward network learning smooth non-linear "
            "relationships via gradient descent; needs more complete records "
            "than tree-based models to fit reliably and gives no feature "
            "importances.",
            [
                "The model converged before the iteration cap.",
                "Hundreds to thousands of complete records were available.",
                "R2 matches or exceeds Random Forest/Gradient Boosting on the same fields.",
            ],
            [
                "Did not converge within the iteration cap.",
                "R2 well below Random Forest Regression on the same fields.",
                "Fewer complete records than roughly ten times the total network weight count.",
            ],
            [
                "Random Forest Regression / Gradient Boosting Regression - usually equal or better on modest planning datasets, easier to interpret.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample R2.",
                "Permutation Feature Importance - the only way to rank fields for an MLP.",
            ],
            "Prefer this tool only when sample size is large and the "
            "relationship is expected to be smoothly non-linear; otherwise a "
            "tree-based model will usually match or beat it with far easier "
            "interpretation.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Neural Network Regression Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>Neural Network Regression (MLP)</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Explanatory fields: <strong>{html.escape(', '.join(feature_fields))}</strong></p>
<p>Hidden layers: <strong>{html.escape(str(hidden_layers))}</strong> | Max iterations: <strong>{max_iter}</strong></p>
<div class="summary">
R2 = {results['r2']:.6f} | RMSE = {results['rmse']:.6f} | MAE = {results['mae']:.6f}<br>
{convergence_note} ({results['n_iter']} iterations used)<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
<h2>Residual vs. Fitted</h2>
{residual_chart}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
