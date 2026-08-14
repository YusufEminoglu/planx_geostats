# -*- coding: utf-8 -*-
"""Extra Trees Classification Processing Algorithm."""
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
from ..core.ml_engines import extract_classification_matrix, fit_extra_trees_classifier, top_feature_importance_rows
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class ExtraTreesClassificationAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
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
        return "extra_trees_classification"

    def displayName(self) -> str:
        return "Extra Trees Classification"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("extra_trees_classification")

    def createInstance(self):
        return ExtraTreesClassificationAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits an Extremely Randomized Trees (Extra Trees) classifier: like "
            "Random Forest Classification, an ensemble voting by averaged class "
            "probability, but each tree also picks its split threshold at random "
            "rather than searching for the locally optimal one. The extra "
            "randomness usually lowers ensemble variance and speeds up training "
            "at a small cost to any single tree's accuracy.\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, and "
            "per-class precision/recall/F1. No out-of-bag score is reported "
            "(Extra Trees does not bootstrap by default) - use Spatial k-Fold "
            "Cross-Validation Evaluator for an honest out-of-sample estimate.\n\n"
            "Run this alongside Random Forest Classification: agreement on "
            "predicted classes and feature importances between the two "
            "algorithms is a useful sanity check that the pattern is real and "
            "not an artifact of one algorithm's particular source of randomness."
        )

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
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output Extra Trees classification layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output Extra Trees HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Extra Trees classification report"))

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
            html_path = os.path.join(tempfile.gettempdir(), "planx_extra_trees_classification_report.html")

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

        feedback.pushInfo(f"Fitting Extra Trees with {n_estimators} trees...")
        try:
            results = fit_extra_trees_classifier(
                x, y, class_labels, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            )
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Extra Trees Classification", ["scikit-learn"], exc))

        feedback.pushInfo(f"Accuracy={results['accuracy']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("et_class", QVariant.String, len=254))
        out_fields.append(QgsField("et_conf", QVariant.Double, len=10, prec=4))
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
                predicted_idx = int(results["fitted"][row_idx])
                out_feature.setAttribute("et_class", class_labels[predicted_idx])
                out_feature.setAttribute("et_conf", float(results["proba"][row_idx][predicted_idx]))
                out_feature.setAttribute("et_used", 1)
            else:
                out_feature.setAttribute("et_class", None)
                out_feature.setAttribute("et_conf", None)
                out_feature.setAttribute("et_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, class_labels, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats Extra Trees classification output",
            {
                "et_class": "Extra Trees predicted class label",
                "et_conf": "Predicted-class probability (0-1); low values mean the ensemble was uncertain",
                "et_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "extra_trees_classification",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, class_labels, results, skipped):
        importance_rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>"
            for name, value in top_feature_importance_rows(feature_fields, results["feature_importances"])
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
            "Extra Trees Classification",
            "An ensemble of decision trees with randomized split thresholds, "
            "voting by averaged class probability.",
            [
                "Random Forest Classification on the same fields gives a comparable accuracy and confusion pattern.",
                "Per-class recall, not just overall accuracy, was checked for imbalanced classes.",
            ],
            [
                "Sharp disagreement with Random Forest Classification's predictions.",
                "A majority class dominates and minority-class recall is near zero.",
            ],
            [
                "Random Forest Classification - cross-check with the bootstrap-based ensemble.",
                "Spatial k-Fold Cross-Validation Evaluator - out-of-sample accuracy (no OOB score here).",
                "SHAP Global Feature Importance - which fields drive the classification.",
            ],
            "Use as a cross-check against Random Forest Classification; treat "
            "et_conf as a confidence surface for field-verification prioritization.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Extra Trees Classification Report</title>
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
<h1>Extra Trees Classification</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Classes: <strong>{html.escape(', '.join(class_labels))}</strong></p>
<div class="summary">
Accuracy = {results['accuracy']:.6f}<br>
Skipped {skipped} record(s) with missing values.
</div>
<h2>Confusion Matrix (rows = actual, columns = predicted)</h2>
<table><thead><tr><th></th>{cm_header}</tr></thead><tbody>{cm_rows}</tbody></table>
<h2>Per-Class Metrics</h2>
<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{per_class_rows}</tbody></table>
<h2>Feature Importance (top 20)</h2>
<table><thead><tr><th>Field</th><th>Importance</th></tr></thead><tbody>{importance_rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
