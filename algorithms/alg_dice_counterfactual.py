# -*- coding: utf-8 -*-
"""DiCE Counterfactual Explanation Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

from ..core.ml_engines import CV_MODEL_KEYS, CV_MODEL_LABELS, extract_classification_matrix, fit_dice_counterfactuals
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class DiCECounterfactualAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    TARGET = "TARGET"
    FEATURES = "FEATURES"
    MODEL = "MODEL"
    FEATURE_ID = "FEATURE_ID"
    DESIRED_CLASS = "DESIRED_CLASS"
    N_COUNTERFACTUALS = "N_COUNTERFACTUALS"
    IMMUTABLE_FEATURES = "IMMUTABLE_FEATURES"
    HTML_REPORT = "HTML_REPORT"

    MODEL_LABELS = [CV_MODEL_LABELS[key] for key in CV_MODEL_KEYS]

    def name(self) -> str:
        return "dice_counterfactual_explanation"

    def displayName(self) -> str:
        return "DiCE Counterfactual Explanation"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("dice_counterfactual_explanation")

    def createInstance(self):
        return DiCECounterfactualAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits the chosen classifier, then searches for the smallest field "
            "changes to one specific record (identified by its QGIS feature "
            "ID) that would flip the model's predicted class to a desired "
            "outcome - Diverse Counterfactual Explanations (Mothilal, Sharma "
            "and Tan, ACM FAT* 2020).\n\n"
            "Where SHAP Local Explanation answers 'what drove this "
            "prediction', DiCE answers the complementary, action-oriented "
            "question: 'what would have to be different for the outcome to "
            "change'. For a parcel classified as unlikely to redevelop, for "
            "example, DiCE reports several distinct, minimal combinations of "
            "field changes (e.g. +12 transit_score, -0.08 pedestrian_ratio) "
            "that the model would instead classify as likely to redevelop - "
            "each a candidate lever, not a forecast.\n\n"
            "Fields listed under Immutable fields are held fixed during the "
            "search (use this for anything the record genuinely cannot "
            "change - a legal zoning code, a fixed lot geometry attribute); "
            "every remaining field is free to vary. Diverse counterfactuals "
            "are requested and returned - several distinct answers rather "
            "than one, since real decisions usually have more than one lever.\n\n"
            "Desired class: leave blank for a binary target (defaults to "
            "'the other class'); for three or more classes, name the exact "
            "target class label.\n\n"
            "Output: an HTML report listing each counterfactual's changed "
            "fields (original value to new value) and the class it would "
            "produce. A counterfactual is only as trustworthy as the "
            "underlying classifier - cross-check its accuracy first with the "
            "Spatial k-Fold Cross-Validation Evaluator.\n\n"
            "Requires the optional dice-ml package (Setup and Diagnostics > "
            "Install / Update GeoStats Libraries)."
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
        self.addParameter(QgsProcessingParameterEnum(self.MODEL, "Model", options=self.MODEL_LABELS, defaultValue=0))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.FEATURE_ID, "Feature ID to explain", type=QgsProcessingParameterNumber.Integer,
                defaultValue=0, minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.DESIRED_CLASS, "Desired class (blank = the other class, binary targets only)", optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_COUNTERFACTUALS, "Number of diverse counterfactuals", type=QgsProcessingParameterNumber.Integer,
                defaultValue=4, minValue=1, maxValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.IMMUTABLE_FEATURES, "Immutable fields (kept fixed; optional)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True, optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output DiCE counterfactual HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "DiCE counterfactual explanation report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        target_field = self.parameterAsString(parameters, self.TARGET, context)
        feature_fields = self.parameterAsFields(parameters, self.FEATURES, context)
        if not feature_fields:
            raise QgsProcessingException("At least one explanatory field must be selected.")

        model_key = CV_MODEL_KEYS[self.parameterAsEnum(parameters, self.MODEL, context)]
        feature_id = self.parameterAsInt(parameters, self.FEATURE_ID, context)
        desired_class = self.parameterAsString(parameters, self.DESIRED_CLASS, context)
        total_cfs = self.parameterAsInt(parameters, self.N_COUNTERFACTUALS, context)
        immutable_features = self.parameterAsFields(parameters, self.IMMUTABLE_FEATURES, context) or []

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_dice_counterfactual_report.html")

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
        if feature_id not in valid_fids:
            raise QgsProcessingException(
                f"Feature ID {feature_id} was not among the complete records used to fit the model "
                "(it may not exist, or it may have missing values in the target/explanatory fields)."
            )
        row_index = valid_fids.index(feature_id)

        if desired_class:
            if desired_class not in class_labels:
                raise QgsProcessingException(f"Class '{desired_class}' not found among: {', '.join(class_labels)}")
            desired_class_index = class_labels.index(desired_class)
        else:
            if len(class_labels) != 2:
                raise QgsProcessingException(
                    "Desired class must be given explicitly when the target has more than 2 classes; "
                    f"classes are: {', '.join(class_labels)}"
                )
            desired_class_index = "opposite"

        invalid_immutable = [name for name in immutable_features if name not in feature_fields]
        if invalid_immutable:
            raise QgsProcessingException(
                f"Immutable field(s) not among the selected explanatory fields: {', '.join(invalid_immutable)}"
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing values.")

        feedback.pushInfo(f"Fitting {CV_MODEL_LABELS[model_key]} and searching for counterfactuals for feature {feature_id}...")
        try:
            results = fit_dice_counterfactuals(
                x, y, class_labels, feature_fields, model_key, row_index, desired_class_index,
                total_cfs=total_cfs, immutable_features=immutable_features,
            )
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("DiCE Counterfactual Explanation", ["dice-ml"], exc))
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        feedback.pushInfo(f"Found {len(results['counterfactuals'])} counterfactual(s).")

        self._write_html(html_path, feature_id, target_field, model_key, class_labels, results)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, feature_id, target_field, model_key, class_labels, results):
        original_class = class_labels[results["original_class_index"]]
        blocks = []
        for i, cf in enumerate(results["counterfactuals"], start=1):
            cf_class = class_labels[cf["class_index"]]
            rows = "".join(
                f"<tr><td>{html.escape(name)}</td><td>{before:.6f}</td><td>{after:.6f}</td>"
                f"<td>{after - before:+.6f}</td></tr>"
                for name, (before, after) in cf["changes"].items()
            )
            if not rows:
                rows = "<tr><td colspan=\"4\">No field changes reported.</td></tr>"
            blocks.append(
                f"<h2>Counterfactual {i} &rarr; {html.escape(cf_class)}</h2>"
                f"<table><thead><tr><th>Field</th><th>Original</th><th>Counterfactual</th><th>Change</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            )
        guidance = analyst_guidance_html(
            "DiCE Counterfactual Explanation",
            "Minimal, diverse field changes to one specific record that would "
            "flip the classifier's predicted class - the action-oriented "
            "complement to SHAP's feature-attribution view.",
            [
                "The classifier's own accuracy was cross-checked (Spatial k-Fold Cross-Validation Evaluator) before trusting its counterfactuals.",
                "The suggested changes are on fields a planner or policy actually has some ability to influence.",
                "Multiple diverse counterfactuals were compared, not just the first one, to see whether they agree on which fields matter.",
            ],
            [
                "Every counterfactual proposes changing a field that cannot realistically change (reconsider the Immutable fields selection).",
                "Counterfactuals require implausibly large changes (may indicate the model has low confidence for this record, or that the desired class is genuinely far from this record's profile).",
                "The underlying classifier's cross-validated accuracy is weak (a counterfactual from an inaccurate model is not a reliable lever).",
            ],
            [
                "SHAP Local Explanation Report - the complementary 'what drove this prediction' view for the same record.",
                "Partial Dependence Report - see whether the counterfactual's suggested field direction matches the model's average behavior across all records.",
                "Spatial k-Fold Cross-Validation Evaluator - honest out-of-sample accuracy for the classifier being explained.",
            ],
            "Present counterfactuals as candidate levers to discuss, not as a "
            "guarantee - they describe what this one model would predict "
            "under a hypothetical change, not a causal effect in the real "
            "planning system.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DiCE Counterfactual Explanation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 8px 10px; text-align: left; font-size: .88rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .7rem; letter-spacing: .05em; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>DiCE Counterfactual Explanation - Feature ID {feature_id}</h1>
<p>Target: <strong>{html.escape(target_field)}</strong> | Model: <strong>{html.escape(CV_MODEL_LABELS[model_key])}</strong></p>
<div class="summary">
Original predicted class = <strong>{html.escape(original_class)}</strong> | Counterfactuals found = {len(results['counterfactuals'])}<br>
Fields free to vary: {html.escape(', '.join(results['features_to_vary']))}
</div>
{''.join(blocks)}
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
