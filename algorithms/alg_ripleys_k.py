# -*- coding: utf-8 -*-
"""Ripley's K-Function Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)

from ..core.analysis_diagnostics import crs_unit_warning
from ..core.stats_engines import calculate_ripleys_k
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class RipleysKFunctionAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    START_DISTANCE = "START_DISTANCE"
    DISTANCE_INCREMENT = "DISTANCE_INCREMENT"
    N_INCREMENTS = "N_INCREMENTS"
    STUDY_AREA = "STUDY_AREA"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "ripleys_k_function"

    def displayName(self) -> str:
        return "Ripley's K-Function"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def icon(self):
        return algorithm_icon("ripleys_k_function")

    def createInstance(self):
        return RipleysKFunctionAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Evaluates point-pattern clustering or dispersion across multiple distance bands "
            "using Ripley's K-Function and the transformed L(d)-d value.\n\n"
            "Positive L(d)-d values indicate more neighbors within distance d than expected "
            "under complete spatial randomness; negative values indicate fewer neighbors. "
            "This implementation uses centroid distances and does not apply edge correction, "
            "so results should be interpreted as a diagnostic planning scan rather than a final "
            "inferential test."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.START_DISTANCE,
                "Starting distance (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=500.0,
                minValue=0.0001,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_INCREMENT,
                "Distance increment (map units)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=500.0,
                minValue=0.0001,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_INCREMENTS,
                "Number of distance increments",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=10,
                minValue=3,
                maxValue=50,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.STUDY_AREA,
                "Study area (optional, map units squared)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Ripley's K-Function report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        start_dist = self.parameterAsDouble(parameters, self.START_DISTANCE, context)
        dist_inc = self.parameterAsDouble(parameters, self.DISTANCE_INCREMENT, context)
        n_inc = self.parameterAsInt(parameters, self.N_INCREMENTS, context)
        study_area_param = self.parameterAsDouble(parameters, self.STUDY_AREA, context)
        study_area = study_area_param if study_area_param > 0 else None

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_ripleys_k.html")

        x_coords = []
        y_coords = []
        skipped = 0
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting feature centroids for Ripley's K-Function...")
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                skipped += 1
                continue
            point = geometry_centroid_point(geom)
            if point is None:
                skipped += 1
                continue
            x_coords.append(point.x())
            y_coords.append(point.y())
            feedback.setProgress(int(25 * (idx / total)))

        if len(x_coords) < 3:
            raise QgsProcessingException("Ripley's K-Function requires at least 3 valid feature geometries.")

        crs_warning = crs_unit_warning(source)
        if crs_warning:
            feedback.pushWarning(crs_warning)
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with empty geometry.")

        results = calculate_ripleys_k(
            np.array(x_coords, dtype=float),
            np.array(y_coords, dtype=float),
            start_dist,
            dist_inc,
            n_inc,
            study_area,
        )
        peak = max(results, key=lambda row: abs(row["l_minus_d"]))
        feedback.pushInfo(
            f"Strongest L(d)-d departure: {peak['l_minus_d']:.4f} at distance {peak['distance']:.2f}."
        )
        self._write_html(html_path, len(x_coords), skipped, results, peak, crs_warning, study_area is not None)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, n, skipped, results, peak, crs_warning, user_area):
        rows = []
        for row in results:
            label = "Peak" if row["distance"] == peak["distance"] else ""
            rows.append(
                "<tr>"
                f"<td class=\"metric-val\">{row['distance']:.2f} {label}</td>"
                f"<td class=\"metric-val\">{row['observed_k']:.6f}</td>"
                f"<td class=\"metric-val\">{row['expected_k']:.6f}</td>"
                f"<td class=\"metric-val\">{row['l_minus_d']:.6f}</td>"
                f"<td class=\"metric-val\">{row['min_neighbors']} / {row['median_neighbors']:.1f} / {row['max_neighbors']}</td>"
                f"<td class=\"metric-val\">{row['isolated_count']}</td>"
                "</tr>"
            )
        if peak["l_minus_d"] > 0:
            pattern = "Clustered relative to complete spatial randomness"
            color = "#e31a1c"
        elif peak["l_minus_d"] < 0:
            pattern = "Dispersed relative to complete spatial randomness"
            color = "#1f78b4"
        else:
            pattern = "Close to complete spatial randomness"
            color = "#718096"

        area_source = "user-provided" if user_area else "layer extent derived"
        next_action = (
            "Use the peak distance as a candidate scale for hot spot, local Moran, or distance-band tools, "
            "then compare the result with planning context and boundary effects."
        )
        crs_block = f"<div class=\"note\"><strong>CRS warning:</strong> {html.escape(crs_warning)}</div>" if crs_warning else ""
        html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Ripley's K-Function</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #25313f; background: #f6f8fb; margin: 0; padding: 24px; }}
.container {{ max-width: 980px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.7rem; }}
h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.interpretation {{ background: #f8fafc; border-left: 5px solid {color}; padding: 16px 18px; margin: 20px 0; }}
.note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
.next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .86rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.metric-val {{ font-family: Consolas, monospace; font-weight: 600; }}
</style>
</head>
<body>
<div class="container">
<h1>Ripley's K-Function</h1>
<p class="subtitle">Features: <strong>{n}</strong> | Skipped: <strong>{skipped}</strong> | Study area: <strong>{results[0]['study_area']:.2f}</strong> ({area_source})</p>
<div class="interpretation"><strong>{pattern}</strong><br>Strongest L(d)-d departure is {peak['l_minus_d']:.6f} at {peak['distance']:.2f} map units.</div>
<h2>Executive Summary</h2>
<p>Ripley's K-Function scans point-pattern behavior across distances. Positive L(d)-d values indicate more neighboring features than expected under complete spatial randomness; negative values indicate fewer. This report is a diagnostic scale scan and does not include edge correction.</p>
{crs_block}
<table>
<thead><tr><th>Distance</th><th>Observed K</th><th>Expected K</th><th>L(d)-d</th><th>Min / Median / Max Neighbors</th><th>Isolated</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<h2>Recommended Next Action</h2>
<div class="next-action">{html.escape(next_action)}</div>
<h2>Assumptions and Caveats</h2>
<ul>
<li>Centroids are used for all input geometries.</li>
<li>No edge correction is applied, so results near study-area boundaries can be biased.</li>
<li>Use a projected CRS and compare plausible study-area boundaries before treating the scale as final.</li>
</ul>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html_doc)
