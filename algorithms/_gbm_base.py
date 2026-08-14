# -*- coding: utf-8 -*-
"""Shared processAlgorithm implementation for the three Gradient Boosting
engines (scikit-learn / XGBoost / LightGBM), regression and classification.

Each concrete alg_gbm_*.py file is a thin subclass that only sets ENGINE and
implements the QGIS-required name()/displayName()/group()/icon()/
createInstance() methods directly (the provider-catalog smoke test parses
each algorithm file's own class body for those, so they cannot live only in
this base class) plus an engine-specific shortHelpString().
"""
from __future__ import annotations

import html
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
from ..core.ml_engines import (
    GBM_ENGINE_LABELS,
    GBM_ENGINE_PACKAGES,
    extract_classification_matrix,
    extract_regression_matrix,
    fit_gbm_classifier,
    fit_gbm_regressor,
    top_feature_importance_rows,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..dependencies import optional_dependency_error


class GBMRegressionBase(HelpUrlMixin, QgsProcessingAlgorithm):
    ENGINE = "sklearn"
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    N_ESTIMATORS = "N_ESTIMATORS"
    LEARNING_RATE = "LEARNING_RATE"
    MAX_DEPTH = "MAX_DEPTH"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

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
                self.N_ESTIMATORS, "Number of boosting rounds", type=QgsProcessingParameterNumber.Integer,
                defaultValue=200, minValue=10, maxValue=3000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.LEARNING_RATE, "Learning rate", type=QgsProcessingParameterNumber.Double,
                defaultValue=0.1, minValue=0.001, maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DEPTH, "Max tree depth (0 = engine default)", type=QgsProcessingParameterNumber.Integer,
                defaultValue=3, minValue=0, maxValue=30,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output Gradient Boosting predictions layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output Gradient Boosting HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Gradient Boosting diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        learning_rate = self.parameterAsDouble(parameters, self.LEARNING_RATE, context)
        max_depth = self.parameterAsInt(parameters, self.MAX_DEPTH, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), f"planx_gbm_{self.ENGINE}_regression_report.html")

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

        feedback.pushInfo(f"Fitting Gradient Boosting ({GBM_ENGINE_LABELS[self.ENGINE]})...")
        try:
            results = fit_gbm_regressor(
                self.ENGINE, x, y, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            )
        except ImportError as exc:
            raise QgsProcessingException(
                optional_dependency_error(self.displayName(), GBM_ENGINE_PACKAGES[self.ENGINE], exc)
            )

        feedback.pushInfo(f"R2={results['r2']:.4f}, RMSE={results['rmse']:.4f}, MAE={results['mae']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("gbm_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("gbm_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("gbm_used", QVariant.Int))

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
                out_feature.setAttribute("gbm_pred", float(results["fitted"][row_idx]))
                out_feature.setAttribute("gbm_resid", float(results["residuals"][row_idx]))
                out_feature.setAttribute("gbm_used", 1)
            else:
                out_feature.setAttribute("gbm_pred", None)
                out_feature.setAttribute("gbm_resid", None)
                out_feature.setAttribute("gbm_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            f"PlanX GeoStats Gradient Boosting ({GBM_ENGINE_LABELS[self.ENGINE]}) regression output",
            {
                "gbm_pred": "Gradient Boosting predicted value",
                "gbm_resid": "Observed minus predicted (positive = model underpredicted)",
                "gbm_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            f"gbm_regression_{self.ENGINE}",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, results, skipped):
        if results["feature_importances"] is None:
            importance_block = (
                "<p>The scikit-learn HistGradientBoosting engine does not expose "
                "feature importances directly - run Permutation Feature Importance "
                "on this fitted model's field selection for a ranked table.</p>"
            )
        else:
            rows = "".join(
                f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>"
                for name, value in top_feature_importance_rows(feature_fields, results["feature_importances"])
            )
            importance_block = (
                f"<h2>Feature Importance (top 20)</h2>"
                f"<table><thead><tr><th>Field</th><th>Importance</th></tr></thead><tbody>{rows}</tbody></table>"
            )
        guidance = analyst_guidance_html(
            f"Gradient Boosting Regression ({GBM_ENGINE_LABELS[self.ENGINE]})",
            "Trees added sequentially, each correcting the previous ensemble's "
            "residual error; typically the strongest tabular-data predictor "
            "among the tree-based tools here, at higher overfitting risk.",
            [
                "R2 improves over Random Forest Regression without a large gap versus a held-out check.",
                "Learning rate and number of boosting rounds were tuned together (lower rate needs more rounds).",
            ],
            [
                "Near-perfect in-sample R2 with a much lower cross-validated score (classic boosting overfit).",
                "Very few complete records relative to the number of boosting rounds.",
            ],
            [
                "Spatial k-Fold Cross-Validation Evaluator - this tool's in-sample R2 is optimistic, always cross-validate boosting models.",
                "Random Forest Regression - a lower-variance baseline to compare against.",
                "SHAP Spatial Attribution Map - explain predictions from the fitted model.",
            ],
            "Treat in-sample R2 here as an upper bound, not an estimate of "
            "real-world accuracy - boosting models fit training data closely "
            "by design.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Gradient Boosting Regression Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .9rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>Gradient Boosting Regression ({html.escape(GBM_ENGINE_LABELS[self.ENGINE])})</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Explanatory fields: <strong>{html.escape(', '.join(feature_fields))}</strong></p>
<div class="summary">
R2 = {results['r2']:.6f} | RMSE = {results['rmse']:.6f} | MAE = {results['mae']:.6f}<br>
Skipped {skipped} record(s) with missing or non-numeric values.
</div>
{importance_block}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)


class GBMClassificationBase(HelpUrlMixin, QgsProcessingAlgorithm):
    ENGINE = "sklearn"
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    N_ESTIMATORS = "N_ESTIMATORS"
    LEARNING_RATE = "LEARNING_RATE"
    MAX_DEPTH = "MAX_DEPTH"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TARGET, "Target field (class labels)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
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
                self.N_ESTIMATORS, "Number of boosting rounds", type=QgsProcessingParameterNumber.Integer,
                defaultValue=200, minValue=10, maxValue=3000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.LEARNING_RATE, "Learning rate", type=QgsProcessingParameterNumber.Double,
                defaultValue=0.1, minValue=0.001, maxValue=1.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_DEPTH, "Max tree depth (0 = engine default)", type=QgsProcessingParameterNumber.Integer,
                defaultValue=3, minValue=0, maxValue=30,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output Gradient Boosting classification layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output Gradient Boosting HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Gradient Boosting classification report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        learning_rate = self.parameterAsDouble(parameters, self.LEARNING_RATE, context)
        max_depth = self.parameterAsInt(parameters, self.MAX_DEPTH, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), f"planx_gbm_{self.ENGINE}_classification_report.html")

        feedback.pushInfo("Extracting complete records...")
        try:
            extraction = extract_classification_matrix(source, feature_fields, target_field, feedback, (0, 30))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y = extraction["x"], extraction["y"]
        class_labels, valid_fids, skipped = extraction["class_labels"], extraction["valid_fids"], extraction["skipped"]
        if len(class_labels) < 2:
            raise QgsProcessingException(
                f"Target field has {len(class_labels)} distinct class(es) among complete records; at least 2 are required."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values.")
        feedback.pushInfo(f"Classes: {', '.join(class_labels)}")

        feedback.pushInfo(f"Fitting Gradient Boosting ({GBM_ENGINE_LABELS[self.ENGINE]})...")
        try:
            results = fit_gbm_classifier(
                self.ENGINE, x, y, class_labels, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
            )
        except ImportError as exc:
            raise QgsProcessingException(
                optional_dependency_error(self.displayName(), GBM_ENGINE_PACKAGES[self.ENGINE], exc)
            )

        feedback.pushInfo(f"Accuracy={results['accuracy']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("gbm_class", QVariant.String, len=254))
        out_fields.append(QgsField("gbm_conf", QVariant.Double, len=10, prec=4))
        out_fields.append(QgsField("gbm_used", QVariant.Int))

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
                predicted_idx = int(results["fitted"][row_idx])
                out_feature.setAttribute("gbm_class", class_labels[predicted_idx])
                out_feature.setAttribute("gbm_conf", float(results["proba"][row_idx][predicted_idx]))
                out_feature.setAttribute("gbm_used", 1)
            else:
                out_feature.setAttribute("gbm_class", None)
                out_feature.setAttribute("gbm_conf", None)
                out_feature.setAttribute("gbm_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, class_labels, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            f"PlanX GeoStats Gradient Boosting ({GBM_ENGINE_LABELS[self.ENGINE]}) classification output",
            {
                "gbm_class": "Gradient Boosting predicted class label",
                "gbm_conf": "Predicted-class probability (0-1)",
                "gbm_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            f"gbm_classification_{self.ENGINE}",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, class_labels, results, skipped):
        if results["feature_importances"] is None:
            importance_block = (
                "<p>The scikit-learn HistGradientBoosting engine does not expose "
                "feature importances directly - run Permutation Feature Importance "
                "on this fitted model's field selection for a ranked table.</p>"
            )
        else:
            rows = "".join(
                f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>"
                for name, value in top_feature_importance_rows(feature_fields, results["feature_importances"])
            )
            importance_block = (
                f"<h2>Feature Importance (top 20)</h2>"
                f"<table><thead><tr><th>Field</th><th>Importance</th></tr></thead><tbody>{rows}</tbody></table>"
            )
        per_class_rows = "".join(
            f"<tr><td>{html.escape(row['label'])}</td><td>{row['precision']:.4f}</td>"
            f"<td>{row['recall']:.4f}</td><td>{row['f1']:.4f}</td><td>{row['support']}</td></tr>"
            for row in results["per_class"]
        )
        cm_header = "".join(f"<th>{html.escape(label)}</th>" for label in class_labels)
        cm_rows = "".join(
            f"<tr><td><strong>{html.escape(class_labels[i])}</strong></td>"
            + "".join(f"<td>{value}</td>" for value in row)
            + "</tr>"
            for i, row in enumerate(results["confusion_matrix"])
        )
        guidance = analyst_guidance_html(
            f"Gradient Boosting Classification ({GBM_ENGINE_LABELS[self.ENGINE]})",
            "Trees added sequentially to correct the previous ensemble's "
            "misclassifications; typically strong accuracy at higher overfitting "
            "risk than Random Forest.",
            [
                "Accuracy improves over Random Forest Classification without a large gap versus a held-out check.",
                "Per-class recall was checked, not just overall accuracy, for imbalanced classes.",
            ],
            [
                "Near-perfect in-sample accuracy (classic boosting overfit signal).",
                "A majority class dominates and minority-class recall is near zero.",
            ],
            [
                "Spatial k-Fold Cross-Validation Evaluator - always cross-validate boosting models before trusting accuracy.",
                "Random Forest Classification - a lower-variance baseline to compare against.",
                "SHAP Global Feature Importance - explain predictions from the fitted model.",
            ],
            "Treat in-sample accuracy here as an upper bound, not an estimate of "
            "real-world accuracy - boosting models fit training data closely by design.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Gradient Boosting Classification Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .88rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .7rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>Gradient Boosting Classification ({html.escape(GBM_ENGINE_LABELS[self.ENGINE])})</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Classes: <strong>{html.escape(', '.join(class_labels))}</strong></p>
<div class="summary">
Accuracy = {results['accuracy']:.6f}<br>
Skipped {skipped} record(s) with missing values.
</div>
<h2>Confusion Matrix (rows = actual, columns = predicted)</h2>
<table><thead><tr><th></th>{cm_header}</tr></thead><tbody>{cm_rows}</tbody></table>
<h2>Per-Class Metrics</h2>
<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{per_class_rows}</tbody></table>
{importance_block}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
