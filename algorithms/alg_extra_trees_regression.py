# -*- coding: utf-8 -*-
"""Extra Trees Regression Processing Algorithm."""
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
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.symbology import apply_renderer, diverging_residual_renderer
from ..core.ml_engines import extract_regression_matrix, fit_extra_trees_regressor, top_feature_importance_rows
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, scatter_plot_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class ExtraTreesRegressionAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    N_ESTIMATORS = "N_ESTIMATORS"
    MAX_DEPTH = "MAX_DEPTH"
    MIN_SAMPLES_LEAF = "MIN_SAMPLES_LEAF"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "extra_trees_regression"

    def displayName(self) -> str:
        return "Extra Trees Regression"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("extra_trees_regression")

    def createInstance(self):
        return ExtraTreesRegressionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits an Extremely Randomized Trees (Extra Trees) regressor: like "
            "Random Forest Regression, an ensemble of decision trees averaged "
            "together, but each tree also picks its split threshold at random "
            "instead of searching for the locally optimal one. This extra "
            "randomness usually makes each individual tree slightly weaker but "
            "reduces the ensemble's variance and training time, which tends to "
            "help on noisy planning data with many correlated explanatory fields.\n\n"
            "Output: fitted values and residuals per complete record, plus an HTML "
            "report with R2, RMSE, MAE, and a ranked feature-importance table. "
            "Extra Trees does not use bootstrap resampling by default, so no "
            "out-of-bag score is reported here - use Spatial k-Fold Cross-Validation "
            "Evaluator for an honest out-of-sample estimate.\n\n"
            "Run this alongside Random Forest Regression on the same fields: if the "
            "two models rank feature importances similarly and produce comparable "
            "R2, the pattern is likely real rather than an artifact of one "
            "algorithm's particular randomness. If they disagree sharply, treat "
            "both results cautiously and check for near-duplicate or highly "
            "correlated explanatory fields."
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
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ESTIMATORS, "Number of trees", type=QgsProcessingParameterNumber.Integer,
                defaultValue=200, minValue=10, maxValue=2000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DEPTH, "Max tree depth (0 = unlimited)", type=QgsProcessingParameterNumber.Integer,
                defaultValue=0, minValue=0, maxValue=100,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_SAMPLES_LEAF, "Minimum samples per leaf", type=QgsProcessingParameterNumber.Integer,
                defaultValue=1, minValue=1, maxValue=200,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output Extra Trees predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output Extra Trees HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Extra Trees diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        max_depth = self.parameterAsInt(parameters, self.MAX_DEPTH, context)
        min_samples_leaf = self.parameterAsInt(parameters, self.MIN_SAMPLES_LEAF, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_extra_trees_regression_report.html")

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

        feedback.pushInfo(f"Fitting Extra Trees with {n_estimators} trees...")
        try:
            results = fit_extra_trees_regressor(
                x, y, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            )
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Extra Trees Regression", ["scikit-learn"], exc))

        feedback.pushInfo(f"R2={results['r2']:.4f}, RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("et_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("et_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("et_used", QVariant.Int))

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
                out_feature.setAttribute("et_pred", float(results["fitted"][row_idx]))
                out_feature.setAttribute("et_resid", float(results["residuals"][row_idx]))
                out_feature.setAttribute("et_used", 1)
            else:
                out_feature.setAttribute("et_pred", None)
                out_feature.setAttribute("et_resid", None)
                out_feature.setAttribute("et_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats Extra Trees regression output",
            {
                "et_pred": "Extra Trees predicted value (mean of all tree predictions)",
                "et_resid": "Observed minus predicted (positive = model underpredicted)",
                "et_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "extra_trees_regression",
        )
        apply_renderer(layer, diverging_residual_renderer(layer, layer.geometryType(), "et_resid"))
        return {}

    def _write_html(self, path, target_field, feature_fields, results, skipped):
        importance_rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>"
            for name, value in top_feature_importance_rows(feature_fields, results["feature_importances"])
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
            "Extra Trees Regression",
            "An ensemble of decision trees with randomized split thresholds; "
            "usually lower variance than Random Forest at similar bias.",
            [
                "Random Forest Regression on the same fields gives a comparable R2 and importance ranking.",
                "At least a few hundred complete records were available.",
            ],
            [
                "Sharp disagreement with Random Forest Regression's importances.",
                "Very few complete records after skipping missing values.",
            ],
            [
                "Random Forest Regression - cross-check with the bootstrap-based ensemble.",
                "Spatial k-Fold Cross-Validation Evaluator - out-of-sample R2 (no OOB score here).",
                "SHAP Spatial Attribution Map - map where each feature matters.",
            ],
            "Use as a cross-check against Random Forest Regression rather than in "
            "isolation; agreement between the two increases confidence in the "
            "explanatory-field ranking.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Extra Trees Regression Report</title>
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
<h1>Extra Trees Regression</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Explanatory fields: <strong>{html.escape(', '.join(feature_fields))}</strong></p>
<div class="summary">
R2 = {results['r2']:.6f} | RMSE = {results['rmse']:.6f} | MAE = {results['mae']:.6f}<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
<h2>Feature Importance (top 20)</h2>
<table><thead><tr><th>Field</th><th>Importance</th></tr></thead><tbody>{importance_rows}</tbody></table>
<h2>Residual vs. Fitted</h2>
{residual_chart}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
