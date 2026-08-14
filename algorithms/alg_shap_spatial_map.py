# -*- coding: utf-8 -*-
"""SHAP Spatial Attribution Map Processing Algorithm."""
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
    QgsProcessingParameterString,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.ml_engines import CV_MODEL_KEYS, CV_MODEL_LABELS, extract_classification_matrix, extract_regression_matrix
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.xai_engines import compute_shap_values, shap_global_importance
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class SHAPSpatialMapAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    TASK_TYPE = "TASK_TYPE"
    MODEL = "MODEL"
    CLASS_LABEL = "CLASS_LABEL"
    MAX_ROWS = "MAX_ROWS"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    TASK_TYPES = ["Regression (numeric target)", "Classification (categorical target)"]
    TASK_KEYS = ["regression", "classification"]
    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "shap_spatial_map"

    def displayName(self) -> str:
        return "SHAP Spatial Attribution Map"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("shap_spatial_map")

    def createInstance(self):
        return SHAPSpatialMapAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "The distinctive GeoStats way to use SHAP: instead of only "
            "producing a summary table, this tool writes each explanatory "
            "field's SHAP contribution back onto the map as a new attribute "
            "column per field (shap_<field name>) - so you can symbolize any "
            "of them and see WHERE that field pushes the prediction up or "
            "down, not just how much it matters on average.\n\n"
            "A field's global importance (from SHAP Global Feature Importance) "
            "can hide large spatial variation: a field that matters a lot in "
            "one district and not at all in another averages out to a "
            "moderate importance score, but the spatial map reveals that "
            "pattern directly. This is often the more actionable view for "
            "planning work than any single summary number.\n\n"
            "Output: the input layer plus one shap_<field> column per "
            "explanatory field (only for the sampled/explained records - see "
            "shap_used), and a shap_base column with the model's average "
            "prediction (every record's prediction equals shap_base plus the "
            "sum of its shap_<field> values). Records beyond Max rows to "
            "explain are not sampled and have shap_used = 0 with NULL SHAP "
            "columns.\n\n"
            "For classification, values are for one class at a time - set "
            "Class to explain, or leave blank for the first class in sorted "
            "label order. Symbolize a shap_<field> column with a diverging "
            "color ramp centered on zero to see where that field pushes the "
            "prediction toward versus away from the explained class."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.TARGET, "Target field", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FEATURES, "Explanatory fields (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(QgsProcessingParameterEnum(self.TASK_TYPE, "Task type", options=self.TASK_TYPES, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(self.MODEL, "Model", options=self.MODEL_LABELS, defaultValue=0))
        self.addParameter(
            QgsProcessingParameterString(
                self.CLASS_LABEL, "Class to explain (classification only; blank = first class alphabetically)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ROWS, "Max rows to explain", type=QgsProcessingParameterNumber.Integer,
                defaultValue=200, minValue=20, maxValue=5000,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output SHAP attribution layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output SHAP spatial map HTML summary", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "SHAP spatial attribution summary"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        task_type = self.TASK_KEYS[self.parameterAsEnum(parameters, self.TASK_TYPE, context)]
        model_key = CV_MODEL_KEYS[self.parameterAsEnum(parameters, self.MODEL, context)]
        class_label = self.parameterAsString(parameters, self.CLASS_LABEL, context)
        max_rows = self.parameterAsInt(parameters, self.MAX_ROWS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_shap_spatial_map_report.html")

        feedback.pushInfo("Extracting complete records...")
        try:
            if task_type == "regression":
                extraction = extract_regression_matrix(source, feature_fields, target_field, feedback, (0, 20))
            else:
                extraction = extract_classification_matrix(source, feature_fields, target_field, feedback, (0, 20))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        x, y, valid_fids, skipped = extraction["x"], extraction["y"], extraction["valid_fids"], extraction["skipped"]
        if len(y) <= len(feature_fields) + 1:
            raise QgsProcessingException(
                f"Insufficient complete records ({len(y)}) for {len(feature_fields)} explanatory field(s)."
            )
        class_index = 0
        class_labels = extraction.get("class_labels")
        if task_type == "classification":
            if len(class_labels) < 2:
                raise QgsProcessingException("At least 2 distinct classes are required for classification.")
            if class_label:
                if class_label not in class_labels:
                    raise QgsProcessingException(f"Class '{class_label}' not found among: {', '.join(class_labels)}")
                class_index = class_labels.index(class_label)
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values.")

        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]} and computing SHAP values...")
        try:
            shap_result = compute_shap_values(x, y, task_type, model_key, class_index=class_index, max_rows=max_rows)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("SHAP Spatial Attribution Map", ["shap"], exc))

        shap_values = shap_result["shap_values"]
        base_value = shap_result["base_value"]
        sample_idx = shap_result["sample_idx"]
        explained_fids = [valid_fids[i] for i in sample_idx]
        shap_by_fid = {fid: shap_values[row] for row, fid in enumerate(explained_fids)}
        feedback.pushInfo(f"Explained {len(explained_fids)} of {len(y)} complete records.")

        shap_field_names = [f"shap_{name}" for name in feature_fields]
        out_fields = source.fields()
        for field_name in shap_field_names:
            out_fields.append(QgsField(field_name, QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("shap_base", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("shap_used", QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        total = source.featureCount() or 1
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in shap_by_fid:
                row = shap_by_fid[fid]
                for field_name, value in zip(shap_field_names, row):
                    out_feature.setAttribute(field_name, float(value))
                out_feature.setAttribute("shap_base", float(base_value))
                out_feature.setAttribute("shap_used", 1)
            else:
                for field_name in shap_field_names:
                    out_feature.setAttribute(field_name, None)
                out_feature.setAttribute("shap_base", None)
                out_feature.setAttribute("shap_used", 0)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(20 + 70 * (current / total)))

        importance_rows = shap_global_importance(shap_values, feature_fields)
        explained_class = class_labels[class_index] if task_type == "classification" else None
        self._write_html(
            html_path, target_field, model_key, explained_class, len(explained_fids), len(y), base_value,
            importance_rows, skipped,
        )
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        field_descriptions = {
            "shap_base": "Model's average prediction across explained records (the SHAP baseline)",
            "shap_used": "1 if this record was sampled and explained, 0 otherwise",
        }
        apply_output_metadata(layer, "PlanX GeoStats SHAP spatial attribution output", field_descriptions, "shap_spatial_map")
        return {}

    def _write_html(self, path, target_field, model_key, explained_class, n_explained, n_total, base_value, importance_rows, skipped):
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>" for name, value in importance_rows
        )
        class_note = f" | Class explained: <strong>{html.escape(explained_class)}</strong>" if explained_class else ""
        guidance = analyst_guidance_html(
            "SHAP Spatial Attribution Map",
            "Per-record, per-field SHAP contributions written to the map as "
            "attribute columns (shap_<field>) - the spatial companion to SHAP "
            "Global Feature Importance's single ranking.",
            [
                "shap_used = 1 for most records of interest (raise Max rows to explain if too many were skipped).",
                "Symbolizing shap_<field> with a diverging ramp centered on zero shows a coherent spatial pattern, not scattered noise.",
            ],
            [
                "Most records have shap_used = 0 (increase Max rows to explain, or run on a smaller extent).",
                "A field with high global importance shows no spatial pattern at all when mapped (may indicate the effect is driven by a few outlier records).",
            ],
            [
                "SHAP Local Explanation Report - full record-level breakdown for any single feature you find interesting on the map.",
                "Model Residual Spatial Autocorrelation Check - confirm the fitted model has not left unexplained spatial structure.",
            ],
            "Each record's prediction equals shap_base plus the sum of its "
            "shap_<field> values - map one field at a time rather than trying "
            "to read all of them together.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SHAP Spatial Attribution Map Summary</title>
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
<h1>SHAP Spatial Attribution Map</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong>{class_note}</p>
<div class="summary">
Explained {n_explained} of {n_total} complete records | Base value = {base_value:.6f}<br>
Skipped {skipped} record(s) with missing values.<br>
Open the output layer and symbolize any shap_&lt;field&gt; column to see where that field's contribution is concentrated.
</div>
<h2>Mean |SHAP value| by Field (for reference)</h2>
<table><thead><tr><th>Field</th><th>Mean |SHAP|</th></tr></thead><tbody>{rows}</tbody></table>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
