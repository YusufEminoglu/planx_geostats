# -*- coding: utf-8 -*-
"""Two-Step Floating Catchment Area (2SFCA) Accessibility Processing Algorithm."""
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
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.accessibility_engines import extract_coords_and_value, two_step_fca
from ..core.layer_metadata import apply_output_metadata
from ..core.reporting import analyst_guidance_css, analyst_guidance_html

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class TwoStepFCAAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    DEMAND = "DEMAND"
    DEMAND_FIELD = "DEMAND_FIELD"
    SUPPLY = "SUPPLY"
    SUPPLY_FIELD = "SUPPLY_FIELD"
    THRESHOLD = "THRESHOLD"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "two_step_fca_accessibility"

    def displayName(self) -> str:
        return "Two-Step Floating Catchment Area (2SFCA) Accessibility"

    def group(self) -> str:
        return "08 | Accessibility"

    def groupId(self) -> str:
        return "planx_accessibility"

    def icon(self):
        return algorithm_icon("two_step_fca_accessibility")

    def createInstance(self):
        return TwoStepFCAAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes classic two-step floating catchment area (2SFCA) "
            "accessibility, the standard planning measure for supply-to-"
            "demand ratio under a hard distance cutoff (Luo & Wang 2003). "
            "Step 1: every supply point's provider-to-population ratio is its "
            "capacity divided by the total demand within Catchment radius of "
            "it. Step 2: every demand point's accessibility score is the sum "
            "of the ratios of every supply point within Catchment radius of "
            "it - so a demand point is well-served either by being near a "
            "high-capacity facility or near several less-crowded ones.\n\n"
            "Distance is straight-line (Euclidean) between feature centroids, "
            "not network routing distance - a standard simplification when a "
            "routable street network is not on hand. Demand Field and Supply "
            "layers must share the same CRS as Demand layer; results are in "
            "the CRS's linear units.\n\n"
            "Output: the Demand layer with an access_2sfca field. Leave "
            "Demand field blank to count every demand record as one unit "
            "(useful when the demand layer is already one point per person/"
            "household); set it to a population/count field when demand "
            "records represent areas or aggregated counts.\n\n"
            "A demand point with access_2sfca = 0 has no supply point within "
            "Catchment radius at all - a hard service gap, distinct from a "
            "low-but-nonzero score (served, but by capacity stretched thin "
            "across many nearby demand points). For a continuous, no-hard-"
            "cutoff alternative, see Gravity-Based Accessibility Index."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.DEMAND, "Demand layer (population / origins)", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEMAND_FIELD, "Demand field (numeric; blank = count each record as 1)",
                parentLayerParameterName=self.DEMAND, type=QgsProcessingParameterField.Numeric, optional=True,
            )
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
        self.addParameter(
            QgsProcessingParameterNumber(
                self.THRESHOLD, "Catchment radius", type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0, minValue=0.001,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output demand layer with 2SFCA accessibility"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output 2SFCA HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "2SFCA accessibility report"))

    def processAlgorithm(self, parameters, context, feedback):
        demand_source = self.parameterAsSource(parameters, self.DEMAND, context)
        supply_source = self.parameterAsSource(parameters, self.SUPPLY, context)
        if demand_source is None or supply_source is None:
            raise QgsProcessingException("Invalid demand or supply layer source.")

        demand_field = self.parameterAsString(parameters, self.DEMAND_FIELD, context) or None
        supply_field = self.parameterAsString(parameters, self.SUPPLY_FIELD, context)
        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_2sfca_report.html")

        feedback.pushInfo("Extracting demand and supply centroids...")
        try:
            demand = extract_coords_and_value(demand_source, demand_field, feedback)
            supply = extract_coords_and_value(supply_source, supply_field, feedback)
        except ValueError as exc:
            raise QgsProcessingException(str(exc))

        if len(demand["coords"]) == 0 or len(supply["coords"]) == 0:
            raise QgsProcessingException("Both demand and supply layers must have at least one usable record.")

        feedback.pushInfo(f"Computing 2SFCA (catchment radius={threshold})...")
        results = two_step_fca(demand["coords"], demand["values"], supply["coords"], supply["values"], threshold)
        accessibility = results["accessibility"]
        zero_access = int(np.sum(accessibility == 0))
        feedback.pushInfo(
            f"Accessibility: mean={float(np.mean(accessibility)):.6f}, max={float(np.max(accessibility)):.6f}, "
            f"{zero_access} demand record(s) with zero access."
        )

        out_fields = demand_source.fields()
        out_fields.append(QgsField("access_2sfca", QVariant.Double, len=14, prec=8))

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
            out_feature.setAttribute("access_2sfca", access_map.get(feature.id()))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, demand_field, supply_field, threshold, accessibility, zero_access, demand["skipped"], supply["skipped"])
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats 2SFCA accessibility output",
            {"access_2sfca": "Two-step floating catchment area accessibility score (0 = no supply within catchment radius)"},
            "two_step_fca_accessibility",
        )
        return {}

    def _write_html(self, path, demand_field, supply_field, threshold, accessibility, zero_access, demand_skipped, supply_skipped):
        guidance = analyst_guidance_html(
            "Two-Step Floating Catchment Area (2SFCA)",
            "Supply-to-demand ratio within a hard catchment radius, summed "
            "per demand point across every supply point that can reach it.",
            [
                "Catchment radius matches a realistic service/travel distance for this facility type.",
                "Demand and supply layers share the same CRS and cover a comparable extent.",
            ],
            [
                "A large share of demand records score exactly zero (hard gaps - map access_2sfca to see where).",
                "Catchment radius set far larger than any realistic trip (blurs every distinction between areas).",
            ],
            [
                "Gravity-Based Accessibility Index - a continuous alternative without 2SFCA's hard cutoff.",
                "Nearest-Facility Coverage Gap - the simplest possible check, just distance to the closest facility.",
            ],
            "Zero-access demand records are real service gaps, not just low "
            "scores - prioritize them first in any facility-siting discussion.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>2SFCA Accessibility Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>Two-Step Floating Catchment Area (2SFCA)</h1>
<p>Demand field: <strong>{html.escape(demand_field or '(count = 1 per record)')}</strong> | Supply field: <strong>{html.escape(supply_field)}</strong> | Catchment radius: <strong>{threshold:g}</strong></p>
<div class="summary">
Mean access_2sfca = {float(np.mean(accessibility)):.6f} | Max = {float(np.max(accessibility)):.6f} | Zero-access records = {zero_access}<br>
Skipped {demand_skipped} demand and {supply_skipped} supply record(s) with missing/invalid geometry or values.
</div>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
