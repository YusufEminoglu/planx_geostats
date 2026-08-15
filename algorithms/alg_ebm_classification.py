# -*- coding: utf-8 -*-
"""Explainable Boosting Machine (EBM) Classification Processing Algorithm."""
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
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.ml_engines import extract_classification_matrix, fit_ebm_classifier
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class EBMClassificationAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "ebm_classification"

    def displayName(self) -> str:
        return "Explainable Boosting Machine Classification (EBM)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("ebm_classification")

    def createInstance(self):
        return EBMClassificationAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits an Explainable Boosting Machine (EBM) classifier - the "
            "same glass-box generalized additive model as EBM Regression "
            "(Lou, Caruana and Gehrke, KDD 2012; Microsoft InterpretML), "
            "here predicting a binary class. As with the regression version, "
            "every prediction's per-field contribution can be read exactly "
            "off the fitted model - no post-hoc SHAP-style sampling "
            "approximation.\n\n"
            "This tool is scoped to exactly 2 classes: EBM's per-field "
            "contributions for a binary target are log-odds toward the "
            "second class (in sorted label order), a single well-defined "
            "number per field per record; multiclass EBM produces a "
            "separate contribution set per class, which does not fit this "
            "tool's one-column-per-field export. Use Random Forest "
            "Classification or a Gradient Boosting engine, paired with SHAP, "
            "for classification targets with 3 or more classes.\n\n"
            "Pairwise interaction terms are disabled so every model term "
            "maps 1:1 onto an explanatory field, keeping the exported "
            "per-field columns unambiguous.\n\n"
            "Output: ebm_class, ebm_conf, ebm_intercept (the model's base "
            "log-odds; roughly constant across records), and one "
            "ebm_<field> exact log-odds contribution column per "
            "explanatory field - ebm_intercept plus the sum of the "
            "ebm_<field> columns for a record reconstructs the predicted "
            "log-odds for the second class (apply a sigmoid to recover the "
            "predicted probability).\n\n"
            "Requires the optional interpret package (Setup and Diagnostics "
            "> Install / Update GeoStats Libraries)."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TARGET, "Target field (exactly 2 classes)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FEATURES, "Explanatory fields (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output EBM classification layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output EBM HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "EBM classification report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_ebm_classification_report.html")

        feedback.pushInfo("Extracting complete records...")
        try:
            extraction = extract_classification_matrix(source, feature_fields, target_field, feedback, (0, 30))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y = extraction["x"], extraction["y"]
        class_labels, valid_fids, skipped = extraction["class_labels"], extraction["valid_fids"], extraction["skipped"]
        if len(class_labels) != 2:
            raise QgsProcessingException(
                f"EBM Classification supports exactly 2 classes; the target field has {len(class_labels)} "
                f"among complete records: {', '.join(class_labels)}"
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values.")
        feedback.pushInfo(f"Classes: {', '.join(class_labels)}")

        feedback.pushInfo("Fitting Explainable Boosting Machine (interactions disabled)...")
        try:
            results = fit_ebm_classifier(x, y, class_labels, feature_fields)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Explainable Boosting Machine Classification", ["interpret"], exc))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        feedback.pushInfo(f"Accuracy={results['accuracy']:.4f}")

        contrib_fields = [f"ebm_{name}" for name in feature_fields]
        out_fields = source.fields()
        out_fields.append(QgsField("ebm_class", QVariant.String, len=254))
        out_fields.append(QgsField("ebm_conf", QVariant.Double, len=10, prec=4))
        out_fields.append(QgsField("ebm_intercept", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("ebm_used", QVariant.Int))
        for cname in contrib_fields:
            out_fields.append(QgsField(cname, QVariant.Double, len=12, prec=6))

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
                out_feature.setAttribute("ebm_class", class_labels[predicted_idx])
                out_feature.setAttribute("ebm_conf", float(results["proba"][row_idx][predicted_idx]))
                out_feature.setAttribute("ebm_intercept", results["intercept"])
                out_feature.setAttribute("ebm_used", 1)
                for col_idx, cname in enumerate(contrib_fields):
                    out_feature.setAttribute(cname, float(results["contributions"][row_idx, col_idx]))
            else:
                out_feature.setAttribute("ebm_class", None)
                out_feature.setAttribute("ebm_conf", None)
                out_feature.setAttribute("ebm_intercept", None)
                out_feature.setAttribute("ebm_used", 0)
                for cname in contrib_fields:
                    out_feature.setAttribute(cname, None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, class_labels, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        field_descriptions = {
            "ebm_class": "EBM predicted class label",
            "ebm_conf": "Predicted-class probability (0-1)",
            "ebm_intercept": "Model base log-odds toward the second class (roughly constant across records)",
            "ebm_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
        }
        apply_output_metadata(
            layer,
            "PlanX GeoStats Explainable Boosting Machine (EBM) classification output",
            field_descriptions,
            "ebm_classification",
        )
        return {}

    def _write_html(self, path, target_field, feature_fields, class_labels, results, skipped):
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
        mean_abs_contrib = [
            (name, float(abs(results["contributions"][:, i]).mean()))
            for i, name in enumerate(feature_fields)
        ]
        mean_abs_contrib.sort(key=lambda item: item[1], reverse=True)
        contrib_rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>"
            for name, value in mean_abs_contrib
        )
        guidance = analyst_guidance_html(
            "Explainable Boosting Machine (EBM) Classification",
            "A fully additive, glass-box boosted classifier whose per-field "
            "log-odds contribution to every prediction is exact, not a "
            "post-hoc approximation - scoped to binary targets.",
            [
                "Accuracy is comparable to Random Forest Classification / Gradient Boosting Classification on the same fields.",
                "ebm_intercept plus the sum of the ebm_<field> columns for a record, passed through a sigmoid, reconstructs ebm_conf for the second class.",
                "The mean |contribution| ranking below makes planning sense for the fields involved.",
            ],
            [
                "Accuracy is notably lower than Gradient Boosting Classification on the same fields (the disabled pairwise interactions may matter here).",
                "A majority class dominates and minority-class recall is near zero.",
            ],
            [
                "Gradient Boosting Classification (any engine) - compare accuracy against a model that does use interactions.",
                "SHAP Global Feature Importance - a black-box-model comparison point using the approximated approach EBM avoids.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample accuracy.",
            ],
            "Prefer EBM when the audience needs the exact reason behind each "
            "classification, not just an approximation - map an ebm_<field> "
            "column directly instead of running a separate SHAP tool.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>EBM Classification Report</title>
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
<h1>Explainable Boosting Machine Classification (EBM)</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Classes: <strong>{html.escape(', '.join(class_labels))}</strong></p>
<div class="summary">
Accuracy = {results['accuracy']:.6f} | Intercept (log-odds) = {results['intercept']:.6f}<br>
Skipped {skipped} record(s) with missing values.
</div>
<h2>Confusion Matrix (rows = actual, columns = predicted)</h2>
<table><thead><tr><th></th>{cm_header}</tr></thead><tbody>{cm_rows}</tbody></table>
<h2>Per-Class Metrics</h2>
<table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>{per_class_rows}</tbody></table>
<h2>Mean |Contribution| by Field (exact, not sampled)</h2>
<table><thead><tr><th>Field</th><th>Mean |ebm log-odds contribution|</th></tr></thead><tbody>{contrib_rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
