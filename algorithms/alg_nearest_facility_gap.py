# -*- coding: utf-8 -*-
"""Nearest-Facility Coverage Gap Processing Algorithm."""
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
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.accessibility_engines import extract_coords_and_value, nearest_facility_gap
from ..core.layer_metadata import apply_output_metadata
from ..core.reporting import analyst_guidance_css, analyst_guidance_html

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class NearestFacilityGapAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    DEMAND = "DEMAND"
    SUPPLY = "SUPPLY"
    THRESHOLD = "THRESHOLD"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "nearest_facility_coverage_gap"

    def displayName(self) -> str:
        return "Nearest-Facility Coverage Gap"

    def group(self) -> str:
        return "08 | Accessibility"

    def groupId(self) -> str:
        return "planx_accessibility"

    def icon(self):
        return algorithm_icon("nearest_facility_coverage_gap")

    def createInstance(self):
        return NearestFacilityGapAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "The simplest accessibility check: for every demand record, the "
            "straight-line distance to the single nearest supply record, and "
            "whether that distance is within Coverage threshold. No capacity "
            "or population weighting - just proximity to the closest "
            "facility of this type, regardless of that facility's size or "
            "how many other demand points also rely on it.\n\n"
            "Output: the Demand layer with nearest_dist (distance to the "
            "closest supply record), covered (1 if within threshold, 0 "
            "otherwise), and nearest_supply_fid (the supply layer's feature "
            "ID of that closest record, so you can trace which facility is "
            "responsible for coverage or the lack of it).\n\n"
            "Use this as a fast first look before running 2SFCA or Gravity-"
            "Based Accessibility Index - it answers 'is there anything "
            "nearby at all' without needing a capacity field, which makes it "
            "usable even for a quick facility-location sanity check before "
            "capacity data has been collected. Because it ignores capacity, "
            "a demand point can show covered = 1 while still being poorly "
            "served in practice if its nearest facility is small and shared "
            "by many other nearby demand points - confirm with 2SFCA before "
            "concluding an area is adequately served."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.DEMAND, "Demand layer (population / origins)", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.SUPPLY, "Supply layer (facilities / destinations)", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.THRESHOLD, "Coverage threshold distance", type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0, minValue=0.001,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output demand layer with coverage gap"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output coverage gap HTML report", fileFilter="HTML files (*.html)", optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Nearest-facility coverage gap report"))

    def processAlgorithm(self, parameters, context, feedback):
        demand_source = self.parameterAsSource(parameters, self.DEMAND, context)
        supply_source = self.parameterAsSource(parameters, self.SUPPLY, context)
        if demand_source is None or supply_source is None:
            raise QgsProcessingException("Invalid demand or supply layer source.")

        threshold = self.parameterAsDouble(parameters, self.THRESHOLD, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_nearest_facility_gap_report.html")

        feedback.pushInfo("Extracting demand and supply centroids...")
        demand = extract_coords_and_value(demand_source, None, feedback)
        supply = extract_coords_and_value(supply_source, None, feedback)
        if len(demand["coords"]) == 0 or len(supply["coords"]) == 0:
            raise QgsProcessingException("Both demand and supply layers must have at least one usable record.")

        feedback.pushInfo(f"Computing nearest-facility distances (threshold={threshold})...")
        results = nearest_facility_gap(demand["coords"], supply["coords"], threshold)
        covered = results["covered"]
        uncovered_count = int(np.sum(~covered))
        feedback.pushInfo(
            f"{uncovered_count} of {len(covered)} demand record(s) are uncovered "
            f"(nearest facility beyond {threshold})."
        )

        out_fields = demand_source.fields()
        out_fields.append(QgsField("nearest_dist", QVariant.Double, len=14, prec=4))
        out_fields.append(QgsField("covered", QVariant.Int))
        out_fields.append(QgsField("nearest_supply_fid", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, out_fields, demand_source.wkbType(), demand_source.sourceCrs()
        )
        self.out_layer_id = dest_id
        supply_fids = supply["valid_fids"]
        result_map = {
            fid: (float(dist), bool(cov), int(supply_fids[idx]))
            for fid, dist, cov, idx in zip(demand["valid_fids"], results["nearest_dist"], covered, results["nearest_idx"])
        }
        total = demand_source.featureCount() or 1
        for current, feature in enumerate(demand_source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                dist, cov, supply_fid = result_map[fid]
                out_feature.setAttribute("nearest_dist", dist)
                out_feature.setAttribute("covered", 1 if cov else 0)
                out_feature.setAttribute("nearest_supply_fid", supply_fid)
            else:
                out_feature.setAttribute("nearest_dist", None)
                out_feature.setAttribute("covered", None)
                out_feature.setAttribute("nearest_supply_fid", None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        self._write_html(html_path, threshold, uncovered_count, len(covered), demand["skipped"], supply["skipped"])
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats nearest-facility coverage gap output",
            {
                "nearest_dist": "Straight-line distance to the nearest supply record",
                "covered": "1 if the nearest supply record is within the coverage threshold, 0 otherwise",
                "nearest_supply_fid": "Feature ID of the nearest supply record, in the supply layer",
            },
            "nearest_facility_coverage_gap",
        )
        return {}

    def _write_html(self, path, threshold, uncovered_count, total_demand, demand_skipped, supply_skipped):
        pct = 100.0 * uncovered_count / total_demand if total_demand else 0.0
        guidance = analyst_guidance_html(
            "Nearest-Facility Coverage Gap",
            "Straight-line distance from every demand point to its single "
            "nearest supply point, with a simple within/beyond threshold flag.",
            [
                "Coverage threshold reflects a realistic maximum acceptable distance for this facility type.",
                "covered = 0 records are being treated as genuine service gaps, not measurement noise.",
            ],
            [
                "A large share of demand records are uncovered (facility siting may be genuinely inadequate).",
                "covered = 1 is being read as 'well served' without checking capacity via 2SFCA.",
            ],
            [
                "Two-Step Floating Catchment Area (2SFCA) - adds capacity/population weighting.",
                "Gravity-Based Accessibility Index - a continuous, no-hard-cutoff view of the same question.",
            ],
            f"{uncovered_count} of {total_demand} demand records ({pct:.1f}%) "
            "have no supply record within the threshold - map covered to see "
            "where the gaps are concentrated.",
        )
        content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Nearest-Facility Coverage Gap Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243040; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 14px 18px; margin: 18px 0; }}
{analyst_guidance_css()}
</style></head>
<body><div class="container">
<h1>Nearest-Facility Coverage Gap</h1>
<p>Coverage threshold: <strong>{threshold:g}</strong></p>
<div class="summary">
Uncovered = {uncovered_count} of {total_demand} demand record(s) ({pct:.1f}%)<br>
Skipped {demand_skipped} demand and {supply_skipped} supply record(s) with missing/invalid geometry.
</div>
{guidance}
</div></body></html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
