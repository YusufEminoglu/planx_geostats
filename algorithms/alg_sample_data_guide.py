# -*- coding: utf-8 -*-
"""Sample dataset guide and loader Processing Algorithm."""
from __future__ import annotations

import html
import os
import tempfile

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFileDestination,
    QgsProject,
    QgsVectorLayer,
)


class SampleDataGuideAlgorithm(QgsProcessingAlgorithm):
    LOAD_IN_PROJECT = "LOAD_IN_PROJECT"
    HTML_REPORT = "HTML_REPORT"
    SAMPLE_PATH = "SAMPLE_PATH"

    LAYER_NAME = "planx_geostats_izmir_neighborhoods"

    def name(self) -> str:
        return "sample_dataset_guide"

    def displayName(self) -> str:
        return "Sample Dataset Guide"

    def group(self) -> str:
        return "00 | Setup and Diagnostics"

    def groupId(self) -> str:
        return "planx_setup_diagnostics"

    def createInstance(self):
        return SampleDataGuideAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Opens a concise guide for the bundled PlanX GeoStats Lab sample dataset "
            "and can optionally load the sample neighborhood layer into the current QGIS project.\n\n"
            "The sample GeoPackage uses English snake_case fields and is intended for demos, "
            "manual QA, regression workflow testing, model comparison, and report language review."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOAD_IN_PROJECT,
                "Load sample layer into the current QGIS project",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output sample dataset guide",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Sample dataset guide"))
        self.addOutput(QgsProcessingOutputString(self.SAMPLE_PATH, "Sample GeoPackage path"))

    def processAlgorithm(self, parameters, context, feedback):
        sample_path = self._sample_path()
        if not os.path.exists(sample_path):
            raise QgsProcessingException(f"Bundled sample dataset was not found: {sample_path}")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_geostats_sample_dataset_guide.html")

        should_load = self.parameterAsBoolean(parameters, self.LOAD_IN_PROJECT, context)
        if should_load:
            uri = f"{sample_path}|layername={self.LAYER_NAME}"
            layer = QgsVectorLayer(uri, "PlanX GeoStats Sample - Izmir Neighborhoods", "ogr")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                feedback.pushInfo("Loaded PlanX GeoStats sample layer into the current QGIS project.")
            else:
                feedback.pushWarning("The sample dataset exists, but QGIS could not load the layer through the OGR provider.")

        self._write_html(html_path, sample_path)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path, self.SAMPLE_PATH: sample_path}

    def _sample_path(self) -> str:
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(plugin_dir, "sample_data", "planx_geostats_izmir_neighborhoods.gpkg")

    def _write_html(self, path: str, sample_path: str) -> None:
        recommended = [
            ("Heat pattern scan", "median_heat_island_index", "Global Moran, Gi*, Local Moran, and Incremental Autocorrelation."),
            ("Green cooling model", "median_land_surface_temp_c", "OLS, GLR, GWR, MGWR, Spatial Lag, and Spatial Error with NDVI, parks, canopy, imperviousness, and building form."),
            ("Model audit", "median_land_surface_temp_c", "Run several models, then compare their outputs with Model Comparison Matrix."),
            ("Equity and vulnerability", "senior_65plus_population", "Hot spot and outlier workflows for vulnerable population concentration."),
        ]
        rows = "".join(
            "<tr>"
            f"<td><strong>{html.escape(title)}</strong></td>"
            f"<td><code>{html.escape(field)}</code></td>"
            f"<td>{html.escape(workflow)}</td>"
            "</tr>"
            for title, field, workflow in recommended
        )
        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Sample Dataset Guide</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #25313f; background: #f6f8fb; margin: 0; padding: 24px; line-height: 1.55; }}
.container {{ max-width: 1040px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.72rem; color: #17212f; }}
h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; border-left: 4px solid #0f766e; padding-left: 10px; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.summary {{ background: #ecfdf5; border-left: 5px solid #0f766e; padding: 16px 18px; margin: 20px 0; }}
.path {{ background: #111827; color: #f9fafb; padding: 12px 14px; border-radius: 6px; font-family: Consolas, monospace; overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .88rem; }}
th {{ background: #ccfbf1; color: #115e59; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="container">
<h1>PlanX GeoStats Lab Sample Dataset</h1>
<p class="subtitle">Layer: <strong>{self.LAYER_NAME}</strong> | Geometry: <strong>Polygon</strong> | CRS: <strong>EPSG:5253</strong> | Features: <strong>237</strong></p>
<section class="summary">This curated sample dataset contains Izmir neighborhood polygons with English planning, climate, green-space, network, population, and built-form indicators. Use it as the default development and QA dataset for PlanX GeoStats workflows.</section>
<h2>Sample Path</h2>
<div class="path">{html.escape(sample_path)}</div>
<h2>Recommended Workflows</h2>
<table>
<thead><tr><th>Workflow</th><th>Suggested Field</th><th>Tools</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<h2>Notes</h2>
<ul>
<li>All sample-facing field names are English snake_case names designed for stable Processing models and scripts.</li>
<li>For expensive local models such as MGWR, create a filtered subset during development if a full run is slow.</li>
<li>Use the bundled README in <code>sample_data</code> for the complete field dictionary.</li>
</ul>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
