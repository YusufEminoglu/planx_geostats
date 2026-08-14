# -*- coding: utf-8 -*-
"""Neural Network (MLP) Classification Processing Algorithm."""
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
from ..core.ml_engines import extract_classification_matrix, fit_mlp_classifier
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class MLPClassificationAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
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
        return "mlp_classification"

    def displayName(self) -> str:
        return "Neural Network Classification (MLP)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("mlp_classification")

    def createInstance(self):
        return MLPClassificationAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a Multi-Layer Perceptron (a small feed-forward neural network) "
            "classifier: explanatory fields pass through one or more hidden "
            "layers before a softmax output assigns class probabilities, with "
            "weights learned by gradient descent. Features are standardized "
            "internally - no manual scaling needed.\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, "
            "per-class precision/recall/F1, and the training convergence status. "
            "A model that hits the iteration cap without converging should not "
            "be trusted - raise Maximum iterations and refit, or simplify the "
            "hidden-layer sizes.\n\n"
            "MLPs need substantially more complete records than tree-based "
            "models and provide no feature importances - on typical single-"
            "district or single-city planning datasets, Random Forest "
            "Classification or Gradient Boosting Classification usually match "
            "or beat it with far easier interpretation. Reach for this tool "
            "specifically when you have a large sample and suspect smooth, "
            "highly non-linear class boundaries that tree splits approximate "
            "poorly.\n\n"
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
        self.addParameter(QgsProcessingParameterString(self.HIDDEN_LAYERS, "Hidden layer sizes (comma-separated)", defaultValue="64,32"))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITER, "Maximum training iterations", type=QgsProcessingParameterNumber.Integer,
                defaultValue=1000, minValue=50, maxValue=20000,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output MLP classification layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output MLP HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "MLP classification report"))

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
            html_path = os.path.join(tempfile.gettempdir(), "planx_mlp_classification_report.html")

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

        feedback.pushInfo(f"Training MLP with hidden layers {hidden_layers}...")
        try:
            results = fit_mlp_classifier(x, y, class_labels, hidden_layer_sizes=hidden_layers, max_iter=max_iter)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Neural Network Classification (MLP)", ["scikit-learn"], exc))

        feedback.pushInfo(
            f"Accuracy={results['accuracy']:.4f}, converged={results['converged']} ({results['n_iter']}/{max_iter} iterations)"
        )
        if not results["converged"]:
            feedback.pushWarning("Training hit the maximum iteration cap without converging; results may be unreliable.")

        out_fields = source.fields()
        out_fields.append(QgsField("mlp_class", QVariant.String, len=254))
        out_fields.append(QgsField("mlp_conf", QVariant.Double, len=10, prec=4))
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
                predicted_idx = int(results["fitted"][row_idx])
                out_feature.setAttribute("mlp_class", class_labels[predicted_idx])
                out_feature.setAttribute("mlp_conf", float(results["proba"][row_idx][predicted_idx]))
                out_feature.setAttribute("mlp_used", 1)
            else:
                out_feature.setAttribute("mlp_class", None)
                out_feature.setAttribute("mlp_conf", None)
                out_feature.setAttribute("mlp_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, hidden_layers, max_iter, class_labels, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats Neural Network (MLP) classification output",
            {
                "mlp_class": "MLP predicted class label",
                "mlp_conf": "Predicted-class probability (0-1) from the softmax output layer",
                "mlp_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "mlp_classification",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, hidden_layers, max_iter, class_labels, results, skipped):
        convergence_note = (
            "Converged before the iteration cap." if results["converged"]
            else "<strong>Did not converge</strong> - hit the maximum iteration cap; treat results with caution."
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
            "Neural Network Classification (MLP)",
            "A small feed-forward network learning non-linear class boundaries "
            "via gradient descent; needs more complete records than tree-based "
            "models and gives no feature importances.",
            [
                "The model converged before the iteration cap.",
                "Hundreds to thousands of complete records were available.",
                "Accuracy matches or exceeds Random Forest/Gradient Boosting on the same fields.",
            ],
            [
                "Did not converge within the iteration cap.",
                "Accuracy well below Random Forest Classification on the same fields.",
                "A majority class dominates and minority-class recall is near zero.",
            ],
            [
                "Random Forest Classification / Gradient Boosting Classification - usually equal or better on modest planning datasets.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample accuracy.",
                "Permutation Feature Importance - the only way to rank fields for an MLP.",
            ],
            "Prefer this tool only when sample size is large and class "
            "boundaries are expected to be smoothly non-linear; otherwise a "
            "tree-based model will usually match or beat it with far easier "
            "interpretation.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Neural Network Classification Report</title>
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
<h1>Neural Network Classification (MLP)</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Classes: <strong>{html.escape(', '.join(class_labels))}</strong></p>
<p>Hidden layers: <strong>{html.escape(str(hidden_layers))}</strong> | Max iterations: <strong>{max_iter}</strong></p>
<div class="summary">
Accuracy = {results['accuracy']:.6f}<br>
{convergence_note} ({results['n_iter']} iterations used)<br>
Skipped {skipped} record(s) with missing values.
</div>
<h2>Confusion Matrix (rows = actual, columns = predicted)</h2>
<table><thead><tr><th></th>{cm_header}</tr></thead><tbody>{cm_rows}</tbody></table>
<h2>Per-Class Metrics</h2>
<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{per_class_rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
