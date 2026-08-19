# -*- coding: utf-8 -*-
"""TabPFN Classification (Tabular Foundation Model) Processing Algorithm."""
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
from ..core.ml_engines import extract_classification_matrix, fit_tabpfn_classifier
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, heatmap_table_svg
from ..core.symbology import apply_renderer, categorical_field_renderer
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class TabPFNClassificationAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    MAX_ROWS = 10000
    MAX_FEATURES = 500
    MAX_CLASSES = 10

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "tabpfn_classification"

    def displayName(self) -> str:
        return "TabPFN Classification (Tabular Foundation Model)"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("tabpfn_classification")

    def createInstance(self):
        return TabPFNClassificationAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a zero-shot classification prediction with TabPFN v2 "
            "(Hollmann et al., 'Accurate predictions on small data with a "
            "tabular foundation model', Nature, 2025) - the same pretrained, "
            "in-context-learning transformer used by TabPFN Regression, here "
            "applied to a categorical target. Your entire training table and "
            "the records to classify are passed through the network together "
            "as a single forward pass; there is no per-dataset training loop, "
            "no hyperparameter search, and results are typically ready in "
            "seconds even though no tuning was performed.\n\n"
            "TabPFN's pretrained classification head supports up to 10 "
            "distinct classes, and the engine's overall operating envelope "
            "(enforced here, raising a clear error rather than degrading "
            "silently) is up to 10,000 rows and 500 features. This is the "
            "newest and most experimental engine in the group - on published "
            "benchmarks it matched or beat tuned Gradient Boosting on small-"
            "to-medium tabular problems, but it has far less accumulated "
            "field experience on planning-domain data specifically than "
            "Random Forest Classification or the Gradient Boosting engines.\n\n"
            "Output: predicted class and prediction confidence per complete "
            "record, plus an HTML report with accuracy, confusion matrix, and "
            "per-class precision/recall/F1. TabPFN gives no native feature "
            "importances - pair it with Permutation Feature Importance or "
            "SHAP Global Feature Importance when the explanatory question "
            "matters, and verify out-of-sample behavior with the Spatial "
            "k-Fold Cross-Validation Evaluator before relying on it.\n\n"
            "Requires the optional tabpfn package (Setup and Diagnostics > "
            "Install / Update GeoStats Libraries) and, on first use, an "
            "internet connection.\n\n"
            "First use also needs a one-time TabPFN license acceptance. On "
            "Windows, TabPFN's interactive browser-login prompt cannot "
            "complete inside QGIS's embedded console, so accept the license "
            "once outside QGIS instead: open https://ux.priorlabs.ai/account, "
            "log in or register, accept the license, copy the API key, then "
            "set it as a permanent Windows environment variable named "
            "TABPFN_TOKEN and restart QGIS. After that, TabPFN runs with no "
            "further prompts."
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
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output TabPFN classification layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output TabPFN HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "TabPFN classification report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")
        if len(feature_fields) > self.MAX_FEATURES:
            raise QgsProcessingException(
                f"TabPFN supports up to {self.MAX_FEATURES} features; {len(feature_fields)} were selected."
            )

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_tabpfn_classification_report.html")

        feedback.pushInfo("Extracting complete records...")
        try:
            extraction = extract_classification_matrix(source, feature_fields, target_field, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y = extraction["x"], extraction["y"]
        class_labels, valid_fids, skipped = extraction["class_labels"], extraction["valid_fids"], extraction["skipped"]
        if len(class_labels) < 2:
            raise QgsProcessingException(
                f"Target field has {len(class_labels)} distinct class(es) among complete records; at least 2 are required."
            )
        if len(class_labels) > self.MAX_CLASSES:
            raise QgsProcessingException(
                f"TabPFN's classification head supports up to {self.MAX_CLASSES} classes; "
                f"the target field has {len(class_labels)}."
            )
        if len(y) > self.MAX_ROWS:
            raise QgsProcessingException(
                f"TabPFN supports up to {self.MAX_ROWS:,} records; this layer has {len(y):,} complete "
                "records. Use Random Forest Classification, a Gradient Boosting engine, or another "
                "tool in this group for larger tables."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values.")
        feedback.pushInfo(f"Classes: {', '.join(class_labels)}")

        feedback.pushInfo("Running TabPFN in-context inference (downloading pretrained weights on first use)...")
        try:
            results = fit_tabpfn_classifier(x, y, class_labels)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("TabPFN Classification", ["tabpfn"], exc))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        feedback.pushInfo(f"Accuracy={results['accuracy']:.4f}")

        out_fields = source.fields()
        out_fields.append(QgsField("tabpfn_class", QVariant.String, len=254))
        out_fields.append(QgsField("tabpfn_conf", QVariant.Double, len=10, prec=4))
        out_fields.append(QgsField("tabpfn_used", QVariant.Int))

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
                out_feature.setAttribute("tabpfn_class", class_labels[predicted_idx])
                out_feature.setAttribute("tabpfn_conf", float(results["proba"][row_idx][predicted_idx]))
                out_feature.setAttribute("tabpfn_used", 1)
            else:
                out_feature.setAttribute("tabpfn_class", None)
                out_feature.setAttribute("tabpfn_conf", None)
                out_feature.setAttribute("tabpfn_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        self._write_html(html_path, target_field, feature_fields, class_labels, results, skipped)
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats TabPFN classification output",
            {
                "tabpfn_class": "TabPFN predicted class label",
                "tabpfn_conf": "Predicted-class probability (0-1) from TabPFN's in-context inference",
                "tabpfn_used": "1 if the record had complete target/feature values and was used to fit and predict, 0 otherwise",
            },
            "tabpfn_classification",
        )
        apply_renderer(layer, categorical_field_renderer(layer, layer.geometryType(), "tabpfn_class"))
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
        cm_chart = heatmap_table_svg(class_labels, class_labels, results["confusion_matrix"], cell_format="{:.0f}")
        guidance = analyst_guidance_html(
            "TabPFN Classification",
            "A pretrained tabular foundation model performing in-context "
            "learning (no training loop, no hyperparameters to tune) - the "
            "newest and least field-tested engine in this group.",
            [
                "The class count sits at or below TabPFN's 10-class pretrained ceiling.",
                "Accuracy is cross-checked against Random Forest Classification / Gradient Boosting Classification on the same fields.",
                "Out-of-sample performance was verified with the Spatial k-Fold Cross-Validation Evaluator before use in a planning decision.",
            ],
            [
                "Accuracy is markedly better than every tree-based engine with no clear explanation (worth a second look for target leakage before celebrating).",
                "The dataset sits near the 10,000-row, 500-feature, or 10-class ceiling (results near a hard operating boundary deserve extra scrutiny).",
                "A majority class dominates and minority-class recall is near zero.",
            ],
            [
                "Gradient Boosting Classification (CatBoost / XGBoost / LightGBM / scikit-learn) - the established, thoroughly field-tested comparison point.",
                "Permutation Feature Importance / SHAP Global Feature Importance - TabPFN itself gives no feature ranking.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample accuracy, essential given how little planning-domain track record this engine has.",
            ],
            "Treat TabPFN as a fast, competitive second opinion alongside "
            "Gradient Boosting rather than a default first choice, until it "
            "has accumulated more field experience on planning-specific data.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TabPFN Classification Report</title>
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
<h1>TabPFN Classification (Tabular Foundation Model)</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Classes: <strong>{html.escape(', '.join(class_labels))}</strong></p>
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
