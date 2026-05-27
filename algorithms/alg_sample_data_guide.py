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
    QgsProcessingParameterEnum,
    QgsProcessingParameterFileDestination,
    QgsProject,
    QgsVectorLayer,
)


from ._icons import algorithm_icon


class SampleDataGuideAlgorithm(QgsProcessingAlgorithm):
    LOAD_IN_PROJECT = "LOAD_IN_PROJECT"
    DATASET_TO_LOAD = "DATASET_TO_LOAD"
    HTML_REPORT = "HTML_REPORT"
    SAMPLE_PATH = "SAMPLE_PATH"
    SYNTHETIC_QA_PATH = "SYNTHETIC_QA_PATH"
    LOADED_LAYERS = "LOADED_LAYERS"

    LAYER_NAME = "planx_geostats_izmir_neighborhoods"
    LOAD_OPTIONS = [
        "Izmir planning sample",
        "Synthetic QA fixture",
        "Both datasets",
    ]
    SYNTHETIC_QA_LAYERS = [
        "qa_points_grid",
        "qa_lines_directional",
        "qa_polygons_mini",
        "qa_ols_model_output",
        "qa_glr_model_output",
        "qa_gwr_model_output",
        "qa_sar_model_output",
        "qa_sem_model_output",
        "qa_mgwr_model_output",
    ]

    def name(self) -> str:
        return "sample_dataset_guide"

    def displayName(self) -> str:
        return "Sample Dataset Guide"

    def group(self) -> str:
        return "00 | Setup and Diagnostics"

    def groupId(self) -> str:
        return "planx_setup_diagnostics"

    def icon(self):
        return algorithm_icon("sample_dataset_guide")

    def createInstance(self):
        return SampleDataGuideAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Opens a concise guide for the bundled PlanX GeoStats Lab sample dataset "
            "and can optionally load the planning sample, the synthetic QA fixture, "
            "or both into the current QGIS project.\n\n"
            "The Izmir sample is the default planning demo. The synthetic QA fixture is "
            "a compact developer/manual QA dataset for point, line, polygon, and "
            "model-output workflow checks."
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
            QgsProcessingParameterEnum(
                self.DATASET_TO_LOAD,
                "Dataset layers to load",
                options=self.LOAD_OPTIONS,
                defaultValue=0,
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
        self.addOutput(QgsProcessingOutputString(self.SYNTHETIC_QA_PATH, "Synthetic QA GeoPackage path"))
        self.addOutput(QgsProcessingOutputString(self.LOADED_LAYERS, "Loaded layer names"))

    def processAlgorithm(self, parameters, context, feedback):
        sample_path = self._sample_path()
        if not os.path.exists(sample_path):
            raise QgsProcessingException(f"Bundled sample dataset was not found: {sample_path}")
        synthetic_qa_path = self._synthetic_qa_path()
        if not os.path.exists(synthetic_qa_path):
            raise QgsProcessingException(f"Bundled synthetic QA dataset was not found: {synthetic_qa_path}")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_geostats_sample_dataset_guide.html")

        should_load = self.parameterAsBoolean(parameters, self.LOAD_IN_PROJECT, context)
        load_mode = self.parameterAsEnum(parameters, self.DATASET_TO_LOAD, context)
        loaded_layers = []
        if should_load:
            if load_mode in (0, 2):
                loaded_layers.extend(
                    self._load_layers(
                        sample_path,
                        [(self.LAYER_NAME, "PlanX GeoStats Sample - Izmir Neighborhoods")],
                        feedback,
                    )
                )
            if load_mode in (1, 2):
                loaded_layers.extend(
                    self._load_layers(
                        synthetic_qa_path,
                        [(name, f"PlanX GeoStats QA - {name}") for name in self.SYNTHETIC_QA_LAYERS],
                        feedback,
                    )
                )

        self._write_html(html_path, sample_path, synthetic_qa_path)
        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
            self.SAMPLE_PATH: sample_path,
            self.SYNTHETIC_QA_PATH: synthetic_qa_path,
            self.LOADED_LAYERS: ", ".join(loaded_layers),
        }

    def _sample_path(self) -> str:
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(plugin_dir, "sample_data", "planx_geostats_izmir_neighborhoods.gpkg")

    def _synthetic_qa_path(self) -> str:
        plugin_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(plugin_dir, "sample_data", "planx_geostats_synthetic_qa.gpkg")

    def _load_layers(self, gpkg_path: str, layer_specs: list[tuple[str, str]], feedback) -> list[str]:
        loaded = []
        for layer_name, display_name in layer_specs:
            uri = f"{gpkg_path}|layername={layer_name}"
            layer = QgsVectorLayer(uri, display_name, "ogr")
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                loaded.append(layer_name)
                feedback.pushInfo(f"Loaded sample layer: {layer_name}")
            else:
                feedback.pushWarning(f"Could not load bundled sample layer through the OGR provider: {layer_name}")
        return loaded

    def _write_html(self, path: str, sample_path: str, synthetic_qa_path: str) -> None:
        recommended = [
            ("Heat pattern scan", "median_heat_island_index", "Global Moran, Gi*, Local Moran, and Incremental Autocorrelation."),
            ("Green cooling model", "median_land_surface_temp_c", "OLS, GLR, GWR, MGWR, Spatial Lag, and Spatial Error with NDVI, parks, canopy, imperviousness, and building form."),
            ("Model audit", "median_land_surface_temp_c", "Run several models, then compare their outputs with Model Comparison Matrix."),
            ("Equity and vulnerability", "senior_65plus_population", "Hot spot and outlier workflows for vulnerable population concentration."),
        ]
        qa_layers = [
            ("qa_points_grid", "Point", "ANN, Ripley's K, distance bands, KNN weights, GLR logistic/Poisson, and point-based regression smoke checks."),
            ("qa_lines_directional", "Line / multiline", "Linear Directional Mean, including multipart line handling."),
            ("qa_polygons_mini", "Polygon", "Queen/rook contiguity and compact local-statistics checks."),
            ("qa_ols_model_output", "Point model output", "Model Comparison Matrix detection for OLS residual and standardized-residual fields."),
            ("qa_glr_model_output", "Point model output", "Model Comparison Matrix detection for GLR fitted, residual, and used-record fields."),
            ("qa_gwr_model_output", "Point model output", "Model Comparison Matrix detection for GWR predicted and residual fields."),
            ("qa_sar_model_output", "Point model output", "Model Comparison Matrix detection for SAR predicted, residual, used-record, and standardized-residual fields."),
            ("qa_sem_model_output", "Point model output", "Model Comparison Matrix detection for SEM predicted, residual, used-record, and standardized-residual fields."),
            ("qa_mgwr_model_output", "Point model output", "Model Comparison Matrix detection for MGWR predicted, residual, used-record, and standardized-residual fields."),
        ]
        rows = "".join(
            "<tr>"
            f"<td><strong>{html.escape(title)}</strong></td>"
            f"<td><code>{html.escape(field)}</code></td>"
            f"<td>{html.escape(workflow)}</td>"
            "</tr>"
            for title, field, workflow in recommended
        )
        qa_rows = "".join(
            "<tr>"
            f"<td><code>{html.escape(layer)}</code></td>"
            f"<td>{html.escape(geometry)}</td>"
            f"<td>{html.escape(purpose)}</td>"
            "</tr>"
            for layer, geometry, purpose in qa_layers
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
<h2>Loading Modes</h2>
<table>
<thead><tr><th>Mode</th><th>Loaded layers</th><th>Use when</th></tr></thead>
<tbody>
<tr><td><strong>Izmir planning sample</strong></td><td><code>{self.LAYER_NAME}</code></td><td>You want the default planning demo and regular manual workflow checks.</td></tr>
<tr><td><strong>Synthetic QA fixture</strong></td><td><code>qa_points_grid</code>, <code>qa_lines_directional</code>, <code>qa_polygons_mini</code>, and exact model-output QA layers.</td><td>You want compact developer QA layers for edge-case testing.</td></tr>
<tr><td><strong>Both datasets</strong></td><td>All bundled planning and QA layers.</td><td>You are preparing a full manual regression pass before release.</td></tr>
</tbody>
</table>
<h2>Synthetic QA Fixture</h2>
<p>The synthetic QA GeoPackage is a small developer and manual-testing fixture. It complements the Izmir planning demo with point, line, polygon, and minimal model-output layers for runtime edge cases that are hard to cover with a single polygon sample.</p>
<div class="path">{html.escape(synthetic_qa_path)}</div>
<table>
<thead><tr><th>Layer</th><th>Geometry</th><th>Purpose</th></tr></thead>
<tbody>{qa_rows}</tbody>
</table>
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
