# -*- coding: utf-8 -*-
"""Support Vector Classification Processing Algorithm."""
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
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.ml_engines import extract_classification_matrix, fit_svc
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, heatmap_table_svg
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class SVCAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    KERNEL = "KERNEL"
    C = "C"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    KERNELS = ["Linear", "RBF (Gaussian, default)", "Polynomial (degree 3)"]
    KERNEL_KEYS = ["linear", "rbf", "poly"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "support_vector_classification"

    def displayName(self) -> str:
        return "Support Vector Classification (SVC)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("support_vector_classification")

    def createInstance(self):
        return SVCAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Support Vector Classification model: finds the decision "
            "boundary (or, with the RBF/Polynomial kernel, a non-linear boundary "
            "in an implicit transformed feature space) that separates classes "
            "with the widest possible margin, defined only by the closest "
            "boundary points (the support vectors). Class probabilities are "
            "estimated with an internal cross-validated calibration step, so "
            "prediction confidence is usable but slightly more expensive to "
            "compute than for tree-based classifiers.\n\n"
            "Features are standardized internally before fitting (SVC is "
            "scale-sensitive) - no manual standardization is needed.\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, and "
            "per-class precision/recall/F1. SVC gives no feature importances - "
            "pair it with Random Forest Classification or SHAP-based tools when "
            "the explanatory question (which field drives the class) matters as "
            "much as classification accuracy.\n\n"
            "RBF or Polynomial kernels fit curved boundaries and tend to win when "
            "classes are not linearly separable; Linear is faster and easier to "
            "reason about when they roughly are. Raise C to fit the training "
            "boundary more tightly (risking overfitting near-ambiguous cases), "
            "lower it for a wider-margin, more regularized boundary."
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
        self.addParameter(QgsProcessingParameterEnum(self.KERNEL, "Kernel", options=self.KERNELS, defaultValue=1))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.C, "Regularization (C)", type=QgsProcessingParameterNumber.Double,
                defaultValue=1.0, minValue=0.001, maxValue=1000.0,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output SVC classification layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output SVC HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "SVC classification report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        kernel = self.KERNEL_KEYS[self.parameterAsEnum(parameters, self.KERNEL, context)]
        c_value = self.parameterAsDouble(parameters, self.C, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_svc_report.html")

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

        feedback.pushInfo(f"Fitting SVC ({kernel} kernel)...")
        try:
            results = fit_svc(x, y, class_labels, kernel=kernel, c=c_value)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Support Vector Classification", ["scikit-learn"], exc))

        feedback.pushInfo(f"Accuracy={results['accuracy']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("svc_class", QVariant.String, len=254))
        out_fields.append(QgsField("svc_conf", QVariant.Double, len=10, prec=4))
        out_fields.append(QgsField("svc_used", QVariant.Int))

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
                out_feature.setAttribute("svc_class", class_labels[predicted_idx])
                out_feature.setAttribute("svc_conf", float(results["proba"][row_idx][predicted_idx]))
                out_feature.setAttribute("svc_used", 1)
            else:
                out_feature.setAttribute("svc_class", None)
                out_feature.setAttribute("svc_conf", None)
                out_feature.setAttribute("svc_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, kernel, c_value, class_labels, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats Support Vector Classification output",
            {
                "svc_class": "SVC predicted class label",
                "svc_conf": "Predicted-class probability (0-1) from the calibrated SVC model",
                "svc_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "support_vector_classification",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, kernel, c_value, class_labels, results, skipped):
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
        cm_chart = heatmap_table_svg(class_labels, class_labels, results["confusion_matrix"], cell_format="{:.0f}")
        guidance = analyst_guidance_html(
            "Support Vector Classification",
            "A maximum-margin decision boundary shaped only by the closest "
            "boundary points; strong for cleanly separable classes, gives no "
            "feature-level explanation on its own.",
            [
                "Accuracy is comparable to or better than Random Forest Classification.",
                "Kernel choice matches the expected boundary shape (RBF for curved, Linear for near-linear separation).",
            ],
            [
                "Accuracy much lower than Random Forest Classification on the same fields.",
                "A majority class dominates and minority-class recall is near zero.",
            ],
            [
                "Random Forest Classification - compare accuracy and get feature importances SVC cannot provide.",
                "SHAP Global Feature Importance - explain predictions from any of these models.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample accuracy.",
            ],
            "Use SVC when class separability itself is the question; pair with "
            "a tree-based tool when you also need to explain which field drives "
            "the class assignment.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Support Vector Classification Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .88rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .7rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
{chart_css()}
</style></head>
<body><div class="container">
<h1>Support Vector Classification (SVC)</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Classes: <strong>{html.escape(', '.join(class_labels))}</strong></p>
<p>Kernel: <strong>{html.escape(kernel)}</strong> | C: <strong>{c_value:g}</strong></p>
<div class="summary">
Accuracy = {results['accuracy']:.6f}<br>
Skipped {skipped} record(s) with missing values.
</div>
<h2>Confusion Matrix (rows = actual, columns = predicted)</h2>
{cm_chart}
<table><thead><tr><th></th>{cm_header}</tr></thead><tbody>{cm_rows}</tbody></table>
<h2>Per-Class Metrics</h2>
<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{per_class_rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
