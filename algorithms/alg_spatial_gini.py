# -*- coding: utf-8 -*-
"""Spatial Gini inequality decomposition Processing Algorithm."""
from __future__ import annotations

import csv
import html
import json
import logging
import math
import os
import tempfile

import numpy as np

from qgis.core import (
    NULL,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingOutputString,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)

from ..core.analysis_diagnostics import (
    caveats_html,
    crs_unit_warning,
    diagnostics_html,
    filter_weights_to_valid_ids,
    neighbor_summary,
    numeric_quality_summary,
    push_diagnostics,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.stats_engines import calculate_spatial_gini
from ..core.weights import build_weights_matrix

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class SpatialGiniAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    HTML_REPORT = "HTML_REPORT"
    SUMMARY_CSV = "SUMMARY_CSV"
    SUMMARY_JSON = "SUMMARY_JSON"
    SUMMARY = "SUMMARY"

    def name(self) -> str:
        return "spatial_gini_inequality"

    def displayName(self) -> str:
        return "Spatial Inequality (Gini and Spatial Gini)"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def icon(self):
        return algorithm_icon("spatial_gini_inequality")

    def createInstance(self):
        return SpatialGiniAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Calculates the classic Gini coefficient and a spatial decomposition of the Gini "
            "numerator into neighbor and non-neighbor pair contributions following the Rey and "
            "Smith spatial Gini logic used in PySAL-style inequality workflows.\n\n"
            "Use this for non-negative planning indicators such as income, exposure, service "
            "access, risk burden, population rates, or resource provision. The report includes "
            "classic Gini, neighbor Gini component, non-neighbor Gini component, spatial Gini "
            "share, spatial polarization, pair counts, and optional permutation inference."
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
            QgsProcessingParameterField(
                self.FIELD,
                "Non-negative numeric field to analyze",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / neighbor definition",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERMUTATIONS,
                "Permutation count for spatial Gini inference",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=99,
                minValue=0,
                maxValue=9999,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANDOM_SEED,
                "Random seed for permutations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=42,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output spatial Gini HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.SUMMARY_CSV,
                "Output spatial Gini summary CSV (optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.SUMMARY_JSON,
                "Output spatial Gini summary JSON (optional)",
                fileFilter="JSON files (*.json)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Spatial Gini inequality report"))
        self.addOutput(QgsProcessingOutputString("SUMMARY_CSV_OUT", "Spatial Gini summary CSV path"))
        self.addOutput(QgsProcessingOutputString("SUMMARY_JSON_OUT", "Spatial Gini summary JSON path"))
        self.addOutput(QgsProcessingOutputString(self.SUMMARY, "Spatial Gini summary"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_type = ["queen", "rook", "knn", "distance"][weight_type_idx]
        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        random_seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_spatial_gini_report.html")
        csv_path = self.parameterAsFileOutput(parameters, self.SUMMARY_CSV, context)
        json_path = self.parameterAsFileOutput(parameters, self.SUMMARY_JSON, context)

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Field '{field_name}' not found.")
        field = source.fields().at(field_idx)
        if not field.isNumeric():
            raise QgsProcessingException(f"Field '{field_name}' must be numeric.")

        feedback.pushInfo("Building spatial weights for Gini decomposition...")
        neighbors, _, id_order, _ = build_weights_matrix(
            source,
            weight_type,
            k_neighbors=k_neighbors,
            distance_band=distance_band,
            feedback=feedback,
        )
        if feedback.isCanceled():
            return {}

        values = {}
        negatives = []
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting non-negative inequality values...")
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            value = self._to_float(feature.attribute(field_idx))
            if value is None:
                continue
            if value < 0.0:
                negatives.append(feature.id())
                continue
            values[feature.id()] = value
            feedback.setProgress(int(25 + 25 * (idx / total)))

        if negatives:
            preview = ", ".join(str(fid) for fid in negatives[:8])
            raise QgsProcessingException(
                "Gini coefficients require non-negative values. "
                f"Negative records were found in feature id(s): {preview}."
            )

        valid_id_order = [fid for fid in id_order if fid in values]
        if len(valid_id_order) < 2:
            raise QgsProcessingException("At least 2 valid non-negative numeric records are required.")

        y = np.array([values[fid] for fid in valid_id_order], dtype=float)
        numeric_summary = numeric_quality_summary(source.featureCount(), values, y)
        filtered_neighbors, _, filtered_ids = filter_weights_to_valid_ids(neighbors, valid_id_order)
        neighborhood_summary = neighbor_summary(filtered_neighbors, filtered_ids)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, neighborhood_summary, crs_warning)

        feedback.pushInfo("Calculating classic Gini and spatial Gini decomposition...")
        try:
            result = calculate_spatial_gini(
                y,
                filtered_neighbors,
                filtered_ids,
                permutations=permutations,
                seed=random_seed,
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        result.update({
            "field": field_name,
            "weight_type": weight_type,
            "k_neighbors": int(k_neighbors),
            "distance_band": float(distance_band),
            "random_seed": int(random_seed),
        })
        summary = self._summary_text(result)
        feedback.pushInfo(summary)
        if result.get("p_sim") is not None:
            feedback.pushInfo(
                "Permutation inference: "
                f"expected non-neighbor component={self._fmt(result['expected_non_neighbor_component'])}, "
                f"z={self._fmt(result['z_non_neighbor_component'])}, "
                f"p_sim={self._fmt(result['p_sim'])}."
            )

        feedback.pushInfo("Writing spatial Gini report...")
        self._write_html(html_path, result, numeric_summary, neighborhood_summary, crs_warning)
        if csv_path:
            self._write_csv(csv_path, result)
            feedback.pushInfo(f"Spatial Gini summary CSV written: {csv_path}")
        if json_path:
            self._write_json(json_path, result, numeric_summary, neighborhood_summary, crs_warning)
            feedback.pushInfo(f"Spatial Gini summary JSON written: {json_path}")

        outputs = {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
            "SUMMARY_CSV_OUT": csv_path or "",
            "SUMMARY_JSON_OUT": json_path or "",
            self.SUMMARY: summary,
        }
        if csv_path:
            outputs[self.SUMMARY_CSV] = csv_path
        if json_path:
            outputs[self.SUMMARY_JSON] = json_path
        return outputs

    def _to_float(self, value):
        if value is None or value == NULL or str(value) == "NULL":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return numeric

    def _summary_text(self, result: dict) -> str:
        return (
            f"Classic Gini={self._fmt(result['gini'])}; "
            f"Spatial Gini share={self._fmt(result['spatial_gini'])}; "
            f"neighbor component={self._fmt(result['neighbor_component'])}; "
            f"non-neighbor component={self._fmt(result['non_neighbor_component'])}; "
            f"polarization={self._fmt(result['polarization'])}."
        )

    def _write_html(
        self,
        path: str,
        result: dict,
        numeric_summary: dict,
        neighborhood_summary: dict,
        crs_warning: str,
    ) -> None:
        interpretation = self._interpretation(result, neighborhood_summary)
        p_text = self._fmt(result.get("p_sim"))
        p_detail = (
            f"Permutation p_sim for high non-neighbor inequality is <strong>{p_text}</strong> "
            f"using {result['permutations']} permutation(s)."
            if result.get("p_sim") is not None
            else "Permutation inference was not requested or not available for this topology."
        )
        guidance_html = analyst_guidance_html(
            "Spatial Gini",
            "Spatial Gini decomposes overall inequality into neighboring and non-neighboring pair contributions.",
            [
                "The selected field is non-negative and represents a meaningful amount, rate, burden, or access metric.",
                "The neighbor graph reflects the planning question and does not contain many isolated observations.",
                "The result is interpreted with the scale of neighborhoods, not as a universal inequality truth.",
            ],
            [
                "Negative values, standardized z-scores, or residuals used as the inequality variable.",
                "A fully connected graph where there is no non-neighbor complement to compare.",
                "Large Gini values interpreted without reviewing which places carry the burden.",
            ],
            [
                "Global Moran's I to test spatial autocorrelation of the same variable",
                "Local Moran's I or Gi* to locate clusters and hot/cold spots",
                "Data Readiness Audit to inspect skew, missingness, and rate construction",
            ],
            "Use this as an equity and spatial-structure screen. It tells whether inequality is spatially organized, then local tools should identify where that organization matters.",
        )
        rows = [
            ("Classic Gini", self._fmt(result["gini"])),
            ("Neighbor Gini component", self._fmt(result["neighbor_component"])),
            ("Non-neighbor Gini component", self._fmt(result["non_neighbor_component"])),
            ("Spatial Gini share", self._fmt(result["spatial_gini"])),
            ("Spatial polarization", self._fmt(result["polarization"])),
            ("Neighbor pair count", str(result["neighbor_pair_count"])),
            ("Non-neighbor pair count", str(result["non_neighbor_pair_count"])),
            ("Average neighbor pair difference", self._fmt(result["neighbor_avg_diff"])),
            ("Average non-neighbor pair difference", self._fmt(result["non_neighbor_avg_diff"])),
            ("Expected non-neighbor component", self._fmt(result["expected_non_neighbor_component"])),
            ("Non-neighbor component z", self._fmt(result["z_non_neighbor_component"])),
            ("Permutation p_sim", p_text),
        ]
        metric_rows = "".join(
            "<tr>"
            f"<td class=\"metric-name\">{html.escape(label)}</td>"
            f"<td class=\"metric-val\">{html.escape(value)}</td>"
            "</tr>"
            for label, value in rows
        )
        crs_block = f"<div class=\"note\"><strong>CRS warning:</strong> {html.escape(crs_warning)}</div>" if crs_warning else ""
        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Spatial Gini Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #243142; background: #f5f7fb; margin: 0; padding: 24px; line-height: 1.55; }}
.container {{ max-width: 880px; margin: 0 auto; background: #fff; border: 1px solid #dbe4ef; border-radius: 8px; padding: 28px; }}
header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 18px; margin-bottom: 22px; }}
h1 {{ color: #162231; margin: 0 0 6px; font-size: 1.68rem; }}
h2 {{ color: #1f2937; font-size: 1.12rem; margin: 28px 0 12px; border-left: 4px solid #0f766e; padding-left: 10px; }}
.subtitle {{ color: #607086; margin: 0; }}
.summary {{ background: #ecfdf5; border-left: 5px solid #0f766e; padding: 16px 18px; border-radius: 4px; margin: 18px 0; }}
.note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 18px 0; }}
.method {{ background: #f8fafc; border: 1px solid #d9e2ec; padding: 14px 18px; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px 12px; text-align: left; vertical-align: top; font-size: .86rem; }}
th {{ background: #ccfbf1; color: #115e59; text-transform: uppercase; font-size: .70rem; letter-spacing: .05em; }}
.metric-name {{ font-weight: 650; color: #263241; }}
.metric-val {{ font-family: monospace; font-weight: 650; }}
footer {{ margin-top: 34px; padding-top: 14px; border-top: 1px solid #edf2f7; color: #7a899c; font-size: .82rem; }}
{analyst_guidance_css()}
</style>
</head>
<body>
<div class="container">
<header>
<h1>Spatial Inequality (Gini and Spatial Gini)</h1>
<p class="subtitle">Field: <strong>{html.escape(result['field'])}</strong> | Features: <strong>{result['n']}</strong> | Weights: <strong>{html.escape(result['weight_type'])}</strong></p>
</header>

<section class="summary"><strong>Executive summary:</strong> {html.escape(interpretation)}</section>
{crs_block}

<h2>Gini Decomposition</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>{metric_rows}</tbody>
</table>

{diagnostics_html(numeric_summary, neighborhood_summary, crs_warning)}

<h2>How This Implements Spatial Gini</h2>
<div class="method">
The classic Gini uses the sum of absolute differences over all observation pairs. This tool splits that numerator into pairs that are spatial neighbors and pairs that are not spatial neighbors. The neighbor and non-neighbor components are normalized by the same denominator as the classic Gini, so they add back to the classic Gini. The reported spatial Gini share is the fraction of total inequality carried by non-neighbor pairs. Spatial polarization is the average non-neighbor pair difference divided by the average neighbor pair difference; values above 1 mean distant pairs differ more than neighboring pairs.
</div>

<h2>Permutation Inference</h2>
<div class="note">{p_detail}</div>

{caveats_html("Spatial Gini", neighborhood_summary, numeric_summary)}

{guidance_html}

<footer>Generated by PlanX GeoStats Lab spatial inequality engine. Method family: Rey and Smith spatial Gini decomposition.</footer>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _interpretation(self, result: dict, neighborhood_summary: dict) -> str:
        gini = float(result["gini"])
        spatial_share = float(result["spatial_gini"])
        polarization = result.get("polarization")
        if gini <= 1.0e-12:
            return "The selected field is effectively equal across valid records; no meaningful inequality was detected."
        if neighborhood_summary.get("isolated", 0) > 0:
            return (
                "Inequality is measurable, but the neighbor graph contains isolated observations. "
                "Review the neighbor definition before using the spatial share for decisions."
            )
        if result.get("non_neighbor_pair_count", 0) == 0:
            return (
                "The graph is fully connected, so the classic Gini is available but the non-neighbor "
                "spatial component is not interpretable."
            )
        if polarization is not None and polarization > 1.15:
            return (
                "Distant pairs differ more than neighboring pairs, suggesting that the inequality has "
                f"a spatially organized structure. The non-neighbor share is {spatial_share:.1%}."
            )
        if polarization is not None and polarization < 0.85:
            return (
                "Neighboring pairs differ more than distant pairs, suggesting local contrast or boundary "
                f"effects. The non-neighbor share is {spatial_share:.1%}."
            )
        return (
            "Overall inequality is present, but neighbor and non-neighbor average differences are similar. "
            "The spatial organization of inequality is modest under the selected neighbor definition."
        )

    def _write_csv(self, path: str, result: dict) -> None:
        keys = self._result_keys()
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerow({key: self._csv_value(result.get(key)) for key in keys})

    def _write_json(
        self,
        path: str,
        result: dict,
        numeric_summary: dict,
        neighborhood_summary: dict,
        crs_warning: str,
    ) -> None:
        payload = {
            "schema": "planx_geostats_spatial_gini",
            "schema_version": "1.0",
            "provider": "PlanX GeoStats Lab",
            "result": result,
            "numeric_summary": numeric_summary,
            "neighborhood_summary": neighborhood_summary,
            "crs_warning": crs_warning,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._json_ready(payload), handle, ensure_ascii=False, indent=2)

    def _result_keys(self) -> list[str]:
        return [
            "field",
            "weight_type",
            "n",
            "mean",
            "sum",
            "gini",
            "neighbor_component",
            "non_neighbor_component",
            "spatial_gini",
            "neighbor_share",
            "non_neighbor_share",
            "polarization",
            "neighbor_pair_count",
            "non_neighbor_pair_count",
            "total_pair_count",
            "neighbor_avg_diff",
            "non_neighbor_avg_diff",
            "expected_non_neighbor_component",
            "std_non_neighbor_component",
            "z_non_neighbor_component",
            "p_sim",
            "p_low_sim",
            "polarization_p_sim",
            "permutations",
            "random_seed",
            "k_neighbors",
            "distance_band",
        ]

    def _fmt(self, value) -> str:
        if value is None:
            return "n/a"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(number):
            return "n/a"
        return f"{number:.6g}"

    def _csv_value(self, value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.12g}" if math.isfinite(value) else ""
        return value

    def _json_ready(self, value):
        if isinstance(value, dict):
            return {str(key): self._json_ready(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._json_ready(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_ready(item) for item in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value
