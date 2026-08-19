# -*- coding: utf-8 -*-
"""Prediction Uncertainty Map Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
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
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.symbology import apply_renderer, sequential_quantile_renderer
from ..core.ml_engines import UNCERTAINTY_MODEL_KEYS, extract_regression_matrix, prediction_uncertainty
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, histogram_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class PredictionUncertaintyMapAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    MODEL = "MODEL"
    N_ESTIMATORS = "N_ESTIMATORS"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    MODEL_LABELS = ["Random Forest", "Extra Trees"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "prediction_uncertainty_map"

    def displayName(self) -> str:
        return "Prediction Uncertainty Map"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("prediction_uncertainty_map")

    def createInstance(self):
        return PredictionUncertaintyMapAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Maps how confident a bagging ensemble (Random Forest or Extra "
            "Trees) is in its own predictions, using the spread of individual "
            "trees' predictions around their mean as an uncertainty proxy: "
            "every tree in the forest was trained on a different bootstrap "
            "resample, so where the trees agree closely the model is "
            "confident, and where they disagree widely the prediction should "
            "be treated with caution. This uncertainty measure is available "
            "only for the two bagging ensembles here - Gradient Boosting, "
            "Support Vector Machine, and Neural Network do not have an "
            "equivalent per-member prediction spread.\n\n"
            "Output: the input layer with unc_pred (mean prediction across "
            "trees), unc_std (standard deviation across trees - the "
            "uncertainty measure itself), and unc_low / unc_high (10th/90th "
            "percentile across trees, a rough interval). Symbolize unc_std to "
            "see where the model's confidence varies across the study area.\n\n"
            "High-uncertainty areas are typically places with few similar "
            "training examples nearby in feature space - sparse data, unusual "
            "combinations of explanatory-field values, or genuinely "
            "boundary/transition zones. Treat predictions in those areas as "
            "indicative rather than precise, and prioritize them first if "
            "field verification or additional data collection is an option."
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
        self.addParameter(QgsProcessingParameterEnum(self.MODEL, "Ensemble", options=self.MODEL_LABELS, defaultValue=0))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_ESTIMATORS, "Number of trees", type=QgsProcessingParameterNumber.Integer,
                defaultValue=200, minValue=10, maxValue=2000,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output uncertainty layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output uncertainty HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Prediction uncertainty report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")
        model_key = UNCERTAINTY_MODEL_KEYS[self.parameterAsEnum(parameters, self.MODEL, context)]
        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_prediction_uncertainty_report.html")

        feedback.pushInfo("Extracting complete numeric records...")
        try:
            extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, valid_fids, skipped = extraction["x"], extraction["y"], extraction["valid_fids"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric values.")

        feedback.pushInfo(f"Fitting {self.MODEL_LABELS[UNCERTAINTY_MODEL_KEYS.index(model_key)]} with {n_estimators} trees...")
        try:
            results = prediction_uncertainty(x, y, model_key, n_estimators=n_estimators)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Prediction Uncertainty Map", ["scikit-learn"], exc))

        feedback.pushInfo(f"Mean uncertainty (std across trees) = {float(np.mean(results['std'])):.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("unc_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("unc_std", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("unc_low", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("unc_high", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("unc_used", QVariant.Int))

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
                out_feature.setAttribute("unc_pred", float(results["mean"][row_idx]))
                out_feature.setAttribute("unc_std", float(results["std"][row_idx]))
                out_feature.setAttribute("unc_low", float(results["lower_10"][row_idx]))
                out_feature.setAttribute("unc_high", float(results["upper_90"][row_idx]))
                out_feature.setAttribute("unc_used", 1)
            else:
                out_feature.setAttribute("unc_pred", None)
                out_feature.setAttribute("unc_std", None)
                out_feature.setAttribute("unc_low", None)
                out_feature.setAttribute("unc_high", None)
                out_feature.setAttribute("unc_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        self._write_html(html_path, target_field, model_key, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats prediction uncertainty output",
            {
                "unc_pred": "Mean prediction across all trees in the ensemble",
                "unc_std": "Standard deviation of predictions across trees - the uncertainty measure (higher = less confident)",
                "unc_low": "10th percentile of per-tree predictions",
                "unc_high": "90th percentile of per-tree predictions",
                "unc_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "prediction_uncertainty_map",
        )
        apply_renderer(layer, sequential_quantile_renderer(layer, layer.geometryType(), "unc_std"))
        return {}

    def _write_html(self, path, target_field, model_key, results, skipped):
        std = results["std"]
        threshold = float(np.percentile(std, 90))
        high_uncertainty_count = int(np.sum(std >= threshold))
        std_chart = histogram_svg(std.tolist(), x_label="unc_std")
        guidance = analyst_guidance_html(
            "Prediction Uncertainty Map",
            "Spread of individual tree predictions around their mean, per "
            "record - available only for bagging ensembles (Random Forest, "
            "Extra Trees), where each tree saw a different bootstrap sample.",
            [
                "High-unc_std areas correspond to sparse training data or unusual field-value combinations, not random noise.",
                "unc_pred is used together with unc_std, not on its own, when acting on a prediction.",
            ],
            [
                "unc_std is high nearly everywhere (the model may be poorly specified rather than genuinely uncertain in specific places).",
                "A high-uncertainty area coincides with a known data-quality problem in the source layer.",
            ],
            [
                "SHAP Local Explanation Report on a high-uncertainty record - see which fields the model is unsure about.",
                "Model Residual Spatial Autocorrelation Check - confirm uncertainty is not itself spatially systematic in a way that points to a missing field.",
            ],
            "Prioritize field verification or additional data collection in "
            "the highest unc_std areas before acting on predictions there.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Prediction Uncertainty Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>Prediction Uncertainty Map</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Ensemble: <strong>{html.escape(model_key.replace('_', ' ').title())}</strong></p>
<div class="summary">
Mean unc_std = {float(np.mean(std)):.6f} | 90th percentile unc_std = {threshold:.6f} ({high_uncertainty_count} record(s) at or above it)<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
<h2>Distribution of Prediction Uncertainty (unc_std)</h2>
{std_chart}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
