# -*- coding: utf-8 -*-
"""Gravity-Based Accessibility Index Processing Algorithm."""
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

from ..core.accessibility_engines import extract_coords_and_value, gravity_accessibility
from ..core.layer_metadata import apply_output_metadata
from ..core.reporting import analyst_guidance_css, analyst_guidance_html

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class GravityAccessibilityAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    DEMAND = "DEMAND"
    SUPPLY = "SUPPLY"
    SUPPLY_FIELD = "SUPPLY_FIELD"
    DECAY = "DECAY"
    DECAY_PARAM = "DECAY_PARAM"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    DECAY_OPTIONS = ["Gaussian (bell-shaped falloff)", "Exponential", "Power (inverse-distance)"]
    DECAY_KEYS = ["gaussian", "exponential", "power"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "gravity_accessibility_index"

    def displayName(self) -> str:
        return "Gravity-Based Accessibility Index"

    def group(self) -> str:
        return "08 | Accessibility"

    def groupId(self) -> str:
        return "planx_accessibility"

    def icon(self):
        return algorithm_icon("gravity_accessibility_index")

    def createInstance(self):
        return GravityAccessibilityAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes gravity-model accessibility: every demand point's "
            "score is the sum, over every supply point, of that supply "
            "point's capacity weighted by a distance-decay function of the "
            "straight-line distance between them. Unlike 2SFCA, there is no "
            "hard catchment cutoff - a distant facility still contributes a "
            "little, just much less than a nearby one, which usually "
            "produces a smoother, more realistic accessibility surface than "
            "a binary in/out catchment.\n\n"
            "Distance is straight-line (Euclidean) between feature "
            "centroids, not network routing distance. Demand and Supply "
            "layers must share the same CRS.\n\n"
            "Output: the Demand layer with an access_gravity field (raw "
            "score, comparable only within this run - it has no fixed upper "
            "bound, so compare relative values across the study area rather "
            "than reading any single number as inherently good or bad).\n\n"
            "Decay function controls how fast influence fades with distance: "
            "Gaussian falls off gently near zero then drops sharply past "
            "Decay parameter (interpreted as a characteristic distance, e.g. "
            "500 = influence drops off around 500 map units); Exponential "
            "decays smoothly throughout (Decay parameter is the distance at "
            "which influence falls to about 37%); Power (Decay parameter as "
            "the exponent, typically 1-2) decays slowly at first and then "
            "more steeply, and is undefined at distance zero so co-located "
            "points get full weight by convention.\n\n"
            "Prefer 2SFCA when your planning question is framed around a "
            "specific hard threshold ('within a 10-minute walk'); prefer "
            "this tool when you want a continuous surface without picking "
            "one specific cutoff distance."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.DEMAND, "Demand layer (population / origins)", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.SUPPLY, "Supply layer (facilities / destinations)", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.SUPPLY_FIELD, "Supply capacity field (numeric)",
                parentLayerParameterName=self.SUPPLY, type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(QgsProcessingParameterEnum(self.DECAY, "Decay function", options=self.DECAY_OPTIONS, defaultValue=0))
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DECAY_PARAM, "Decay parameter (characteristic distance, or exponent for Power)",
                type=QgsProcessingParameterNumber.Double, defaultValue=500.0, minValue=0.001,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output demand layer with gravity accessibility"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output gravity accessibility HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Gravity accessibility report"))

    def processAlgorithm(self, parameters, context, feedback):
        demand_source = self.parameterAsSource(parameters, self.DEMAND, context)
        supply_source = self.parameterAsSource(parameters, self.SUPPLY, context)
        if demand_source is None or supply_source is None:
            raise QgsProcessingException("Invalid demand or supply layer source.")

        supply_field = self.parameterAsString(parameters, self.SUPPLY_FIELD, context)
        decay = self.DECAY_KEYS[self.parameterAsEnum(parameters, self.DECAY, context)]
        decay_param = self.parameterAsDouble(parameters, self.DECAY_PARAM, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_gravity_accessibility_report.html")

        feedback.pushInfo("Extracting demand and supply centroids...")
        try:
            demand = extract_coords_and_value(demand_source, None, feedback)
            supply = extract_coords_and_value(supply_source, supply_field, feedback)
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        if len(demand["coords"]) == 0 or len(supply["coords"]) == 0:
            raise QgsProcessingException("Both demand and supply layers must have at least one usable record.")

        feedback.pushInfo(f"Computing gravity accessibility ({decay} decay, parameter={decay_param})...")
        results = gravity_accessibility(demand["coords"], supply["coords"], supply["values"], decay=decay, decay_param=decay_param)
        accessibility = results["accessibility"]
        feedback.pushInfo(
            f"Accessibility: mean={float(np.mean(accessibility)):.6f}, min={float(np.min(accessibility)):.6f}, "
            f"max={float(np.max(accessibility)):.6f}"
        )

        out_fields = demand_source.fields()
        out_fields.append(QgsField("access_gravity", QVariant.Double, len=16, prec=8))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, out_fields, demand_source.wkbType(), demand_source.sourceCrs()
        )
        self.out_layer_id = dest_id
        access_map = {fid: float(value) for fid, value in zip(demand["valid_fids"], accessibility)}
        total = demand_source.featureCount() or 1
        for current, feature in enumerate(demand_source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            out_feature.setAttribute("access_gravity", access_map.get(feature.id()))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, supply_field, decay, decay_param, accessibility, demand["skipped"], supply["skipped"])
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats gravity accessibility output",
            {"access_gravity": "Distance-decay-weighted sum of supply capacity reachable from this demand point (relative score, no fixed scale)"},
            "gravity_accessibility_index",
        )
        return {}

    def _write_html(self, path, supply_field, decay, decay_param, accessibility, demand_skipped, supply_skipped):
        guidance = analyst_guidance_html(
            "Gravity-Based Accessibility Index",
            "Distance-decay-weighted sum of supply capacity, computed per "
            "demand point with no hard catchment cutoff.",
            [
                "Decay parameter reflects a realistic characteristic distance for this facility type and travel mode.",
                "Demand and supply layers share the same CRS and cover a comparable extent.",
            ],
            [
                "Nearly uniform scores everywhere (Decay parameter likely too large relative to the study area).",
                "Extreme scores concentrated at just one or two demand points right next to a large-capacity facility.",
            ],
            [
                "Two-Step Floating Catchment Area (2SFCA) - the hard-cutoff alternative, easier to explain to non-technical audiences.",
                "Nearest-Facility Coverage Gap - simplest possible distance-only check.",
            ],
            "Read access_gravity as a relative ranking across this study "
            "area's demand points, not as an absolute, comparable-across-"
            "studies accessibility score.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Gravity Accessibility Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>Gravity-Based Accessibility Index</h1>
<p>Supply field: <strong>{html.escape(supply_field)}</strong> | Decay: <strong>{html.escape(decay)}</strong> | Parameter: <strong>{decay_param:g}</strong></p>
<div class="summary">
Mean access_gravity = {float(np.mean(accessibility)):.6f} | Min = {float(np.min(accessibility)):.6f} | Max = {float(np.max(accessibility)):.6f}<br>
Skipped {demand_skipped} demand and {supply_skipped} supply record(s) with missing/invalid geometry or values.
</div>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
