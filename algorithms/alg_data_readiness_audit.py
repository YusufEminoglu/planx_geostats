# -*- coding: utf-8 -*-
"""Data readiness audit for PlanX GeoStats Lab workflows."""
from __future__ import annotations

import html
import math
import os
import tempfile
import csv
import json
from typing import Optional

import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingOutputString,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsWkbTypes,
)


class DataReadinessAuditAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    HTML_REPORT = "HTML_REPORT"
    FIELD_AUDIT_CSV = "FIELD_AUDIT_CSV"
    AUDIT_JSON = "AUDIT_JSON"
    SUMMARY = "SUMMARY"

    def name(self) -> str:
        return "data_readiness_audit"

    def displayName(self) -> str:
        return "Data Readiness Audit"

    def group(self) -> str:
        return "00 | Setup and Diagnostics"

    def groupId(self) -> str:
        return "planx_setup_diagnostics"

    def createInstance(self):
        return DataReadinessAuditAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Reviews a vector layer before spatial statistics and modeling. The audit checks CRS risk, "
            "geometry availability, numeric field completeness, missing values, non-finite values, "
            "constant and near-constant indicators, and sample-specific PlanX workflow readiness.\n\n"
            "Use this before Global Moran, hot spot analysis, local outlier analysis, regression, GWR, "
            "MGWR, and spatial autoregression workflows. The output is an English HTML report designed "
            "for analyst QA notes and reproducible project documentation."
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
                self.FIELDS,
                "Numeric fields to audit (optional, blank audits all numeric fields)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output data readiness HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.FIELD_AUDIT_CSV,
                "Output field audit CSV (optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.AUDIT_JSON,
                "Output full audit JSON (optional)",
                fileFilter="JSON files (*.json)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Data readiness audit report"))
        self.addOutput(QgsProcessingOutputString("FIELD_AUDIT_CSV_OUT", "Field audit CSV path"))
        self.addOutput(QgsProcessingOutputString("AUDIT_JSON_OUT", "Full audit JSON path"))
        self.addOutput(QgsProcessingOutputString(self.SUMMARY, "Audit summary"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        selected_fields = self.parameterAsFields(parameters, self.FIELDS, context)
        numeric_fields = self._numeric_field_names(source)
        if selected_fields:
            audit_fields = [name for name in selected_fields if name in numeric_fields]
            missing = [name for name in selected_fields if name not in numeric_fields]
            for name in missing:
                feedback.pushWarning(f"Field '{name}' is not numeric or was not found; it will be skipped.")
        else:
            audit_fields = numeric_fields

        if not audit_fields:
            raise QgsProcessingException("No numeric fields were available for the data readiness audit.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_geostats_data_readiness_audit.html")
        csv_path = self.parameterAsFileOutput(parameters, self.FIELD_AUDIT_CSV, context)
        json_path = self.parameterAsFileOutput(parameters, self.AUDIT_JSON, context)

        feedback.pushInfo(f"Auditing {len(audit_fields)} numeric field(s) for GeoStats readiness.")
        layer_profile = self._layer_profile(source)
        layer_profile.update(self._geometry_diagnostics(source, feedback))
        field_summaries, field_values = self._field_summaries(source, audit_fields, feedback)
        correlation_findings = self._correlation_findings(field_values)
        workflow_findings = self._workflow_findings({item["field"] for item in field_summaries})
        overall = self._overall_assessment(layer_profile, field_summaries, correlation_findings)

        self._push_feedback(feedback, layer_profile, field_summaries, overall)
        self._write_html(html_path, layer_profile, field_summaries, correlation_findings, workflow_findings, overall)
        if csv_path:
            self._write_field_audit_csv(csv_path, field_summaries)
            feedback.pushInfo(f"Field audit CSV written: {csv_path}")
        if json_path:
            self._write_audit_json(json_path, layer_profile, field_summaries, correlation_findings, workflow_findings, overall)
            feedback.pushInfo(f"Full audit JSON written: {json_path}")

        outputs = {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
            "FIELD_AUDIT_CSV_OUT": csv_path or "",
            "AUDIT_JSON_OUT": json_path or "",
            self.SUMMARY: overall["summary"],
        }
        if csv_path:
            outputs[self.FIELD_AUDIT_CSV] = csv_path
        if json_path:
            outputs[self.AUDIT_JSON] = json_path
        return outputs

    def _numeric_field_names(self, source) -> list[str]:
        names = []
        for field in source.fields():
            try:
                if field.isNumeric():
                    names.append(field.name())
            except Exception:
                continue
        return names

    def _layer_profile(self, source) -> dict:
        crs_authid = "Unknown"
        crs_description = "Unknown"
        geographic = False
        try:
            crs = source.sourceCrs()
            crs_authid = crs.authid() or "Unknown"
            crs_description = crs.description() or "Unknown"
            geographic = bool(crs.isGeographic())
        except Exception:
            pass

        geometry_label = "Unknown"
        try:
            geometry_label = QgsWkbTypes.displayString(source.wkbType())
        except Exception:
            pass

        return {
            "feature_count": int(source.featureCount()),
            "field_count": int(len(source.fields())),
            "geometry": geometry_label,
            "crs_authid": crs_authid,
            "crs_description": crs_description,
            "is_geographic": geographic,
        }

    def _geometry_diagnostics(self, source, feedback) -> dict:
        total = int(source.featureCount())
        empty = 0
        invalid = 0
        multipart = 0
        checked_validity = 0

        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                empty += 1
                continue
            try:
                if geom.isMultipart():
                    multipart += 1
            except Exception:
                pass
            try:
                checked_validity += 1
                if not geom.isGeosValid():
                    invalid += 1
            except Exception:
                checked_validity -= 1
            if total:
                feedback.setProgress(int(25 * idx / total))

        return {
            "empty_geometry_count": int(empty),
            "invalid_geometry_count": int(invalid),
            "multipart_geometry_count": int(multipart),
            "validity_checked_count": int(max(0, checked_validity)),
        }

    def _field_summaries(self, source, field_names: list[str], feedback) -> tuple[list[dict], dict[str, list[Optional[float]]]]:
        total = int(source.featureCount())
        values = {name: [] for name in field_names}
        aligned_values = {name: [] for name in field_names}
        missing = {name: 0 for name in field_names}
        non_finite = {name: 0 for name in field_names}

        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            for name in field_names:
                raw = feature.attribute(name)
                if raw is None or raw == QVariant() or str(raw) == "NULL":
                    missing[name] += 1
                    aligned_values[name].append(None)
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    missing[name] += 1
                    aligned_values[name].append(None)
                    continue
                if not math.isfinite(value):
                    non_finite[name] += 1
                    aligned_values[name].append(None)
                    continue
                values[name].append(value)
                aligned_values[name].append(value)
            if total:
                feedback.setProgress(int(100 * idx / total))

        summaries = []
        for name in field_names:
            arr = np.array(values[name], dtype=float)
            valid = int(len(arr))
            miss = int(missing[name])
            nonfinite = int(non_finite[name])
            std = float(np.std(arr)) if valid else None
            mean = float(np.mean(arr)) if valid else None
            unique = int(len(np.unique(arr))) if valid else 0
            missing_pct = (100.0 * (miss + nonfinite) / total) if total else 0.0
            constant = bool(valid > 0 and unique <= 1)
            near_constant = bool(valid > 1 and std is not None and std <= 1e-9)
            readiness = self._field_readiness(total, valid, missing_pct, constant, near_constant)
            role = self._analysis_role(name, valid, total, missing_pct, unique, std)
            summaries.append({
                "field": name,
                "total": total,
                "valid": valid,
                "missing": miss,
                "non_finite": nonfinite,
                "missing_pct": missing_pct,
                "minimum": float(np.min(arr)) if valid else None,
                "maximum": float(np.max(arr)) if valid else None,
                "mean": mean,
                "std": std,
                "unique": unique,
                "constant": constant,
                "near_constant": near_constant,
                "readiness": readiness,
                "analysis_role": role["role"],
                "role_reason": role["reason"],
                "suggested_tools": role["tools"],
            })
        return summaries, aligned_values

    def _analysis_role(self, name: str, valid: int, total: int, missing_pct: float, unique: int, std: Optional[float]) -> dict:
        lower = name.lower()
        if valid < 4 or unique <= 1 or (std is not None and std <= 1e-9):
            return {
                "role": "Do not model yet",
                "reason": "The field has too little valid variation for reliable statistics.",
                "tools": "Repair or replace the indicator before analysis.",
            }

        outcome_terms = [
            "temp", "heat", "risk", "index", "score", "vulnerability", "access", "exposure",
            "population", "density", "crime", "value", "price", "income", "demand",
        ]
        explanatory_terms = [
            "ndvi", "park", "canopy", "impervious", "coverage", "floor", "building", "street",
            "connectivity", "integration", "slope", "distance", "area", "ratio", "pct",
        ]
        count_terms = ["count", "number", "population", "household", "building", "unit"]

        if any(term in lower for term in outcome_terms):
            return {
                "role": "Target or pattern variable",
                "reason": "The field name suggests a planning outcome, exposure, index, score, or population measure.",
                "tools": "Global Moran, Incremental Autocorrelation, Gi*, Local Moran, Bivariate Lee's L, OLS/GLR/GWR/MGWR.",
            }
        if any(term in lower for term in explanatory_terms):
            return {
                "role": "Candidate explanatory variable",
                "reason": "The field name suggests a built-form, green-space, network, or environmental driver.",
                "tools": "Exploratory Regression, OLS, GLR, GWR, MGWR, Spatial Lag, Spatial Error, Similarity Search.",
            }
        if any(term in lower for term in count_terms):
            return {
                "role": "Count or intensity variable",
                "reason": "The field name suggests a count-like indicator; normalize it if the denominator varies spatially.",
                "tools": "Poisson GLR, Gi*, Local Moran, rates after denominator review.",
            }
        if missing_pct <= 5.0 and unique >= max(10, int(valid * 0.15)):
            return {
                "role": "General numeric indicator",
                "reason": "The field has enough valid values and variation for exploratory spatial screening.",
                "tools": "Global Moran, Gi*, Local Moran, Similarity Search, clustering, or model screening.",
            }
        return {
            "role": "Review before analysis",
            "reason": "The field is usable numerically, but its meaning and distribution should be reviewed first.",
            "tools": "Data inspection, histogram review, then targeted GeoStats tools.",
        }

    def _correlation_findings(self, field_values: dict[str, list[Optional[float]]]) -> dict:
        fields = list(field_values.keys())
        pairs = []
        max_abs_correlation = 0.0
        for i, left in enumerate(fields):
            for right in fields[i + 1:]:
                x_vals = []
                y_vals = []
                for x, y in zip(field_values[left], field_values[right]):
                    if x is None or y is None:
                        continue
                    x_vals.append(float(x))
                    y_vals.append(float(y))
                if len(x_vals) < 4:
                    continue
                x_arr = np.array(x_vals, dtype=float)
                y_arr = np.array(y_vals, dtype=float)
                if float(np.std(x_arr)) <= 1e-9 or float(np.std(y_arr)) <= 1e-9:
                    continue
                corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
                if not math.isfinite(corr):
                    continue
                abs_corr = abs(corr)
                max_abs_correlation = max(max_abs_correlation, abs_corr)
                if abs_corr >= 0.85:
                    pairs.append({
                        "left": left,
                        "right": right,
                        "correlation": corr,
                        "abs_correlation": abs_corr,
                        "complete_records": len(x_vals),
                    })
        pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
        return {
            "max_abs_correlation": max_abs_correlation,
            "high_pairs": pairs,
            "audited_pair_count": int(len(fields) * (len(fields) - 1) / 2),
        }

    def _field_readiness(self, total: int, valid: int, missing_pct: float, constant: bool, near_constant: bool) -> str:
        if valid < 4:
            return "Not ready: fewer than four valid numeric records"
        if constant or near_constant:
            return "Not ready: no meaningful numeric variation"
        if total and valid / total < 0.75:
            return "Review: substantial missing or invalid data"
        if missing_pct > 10.0:
            return "Review: missing values may affect interpretation"
        return "Ready for GeoStats workflows"

    def _workflow_findings(self, available_fields: set[str]) -> list[dict]:
        workflows = [
            {
                "name": "Urban heat pattern scan",
                "required": ["median_heat_island_index", "median_land_surface_temp_c"],
                "target": "median_heat_island_index or median_land_surface_temp_c",
                "explanatory": "Use green and built-form fields only after the pattern scan.",
                "tools": "Global Moran, Incremental Autocorrelation, Gi*, Local Moran",
                "sequence": "1. Run Data Readiness Audit. 2. Use Incremental Autocorrelation or Calculate Distance Band to choose scale. 3. Run Global Moran for the global signal. 4. Map Gi* and Local Moran to locate hot spots and spatial outliers.",
                "purpose": "Find whether heat-related indicators cluster, disperse, or form local hot and cold spots.",
            },
            {
                "name": "Green cooling and built-form model",
                "required": ["median_land_surface_temp_c", "median_ndvi", "park_m2_per_capita", "impervious_surface_pct", "building_coverage_pct"],
                "target": "median_land_surface_temp_c",
                "explanatory": "median_ndvi, park_m2_per_capita, impervious_surface_pct, building_coverage_pct",
                "tools": "OLS, GLR, GWR, MGWR, Spatial Lag, Spatial Error, Model Comparison Matrix",
                "sequence": "1. Review multicollinearity pairs. 2. Run Exploratory Regression or OLS. 3. Compare OLS, GLR, GWR/MGWR, Spatial Lag, and Spatial Error. 4. Use Model Comparison Matrix and residual spatial diagnostics.",
                "purpose": "Explain temperature variation using vegetation, public green access, imperviousness, and urban form.",
            },
            {
                "name": "Accessibility and network structure",
                "required": ["street_connectivity", "normalized_integration"],
                "target": "normalized_integration or a planning access score",
                "explanatory": "street_connectivity, normalized_choice, closeness_centrality, betweenness_centrality",
                "tools": "Similarity Search, Multivariate Clustering, regression diagnostics",
                "sequence": "1. Treat low-cardinality network fields as ordinal support indicators. 2. Use Similarity Search to compare neighborhoods. 3. Run Multivariate Clustering for typologies. 4. Model only after checking field meaning and variation.",
                "purpose": "Compare neighborhoods by movement-network indicators and identify similar planning profiles.",
            },
            {
                "name": "Equity and vulnerable population review",
                "required": ["senior_65plus_population", "youth_population", "park_m2_per_capita"],
                "target": "senior_65plus_population, youth_population, or a normalized vulnerability rate",
                "explanatory": "park_m2_per_capita, heat, green-space, accessibility, and service fields",
                "tools": "Gi*, Local Moran, Bivariate Lee's L, Exploratory Regression",
                "sequence": "1. Convert raw counts to rates when denominators vary. 2. Run Gi* and Local Moran for concentration and outlier review. 3. Use Bivariate Lee's L to compare population and environmental indicators. 4. Move to regression only after rate construction is defensible.",
                "purpose": "Audit whether vulnerable population groups align with environmental or service-access disadvantages.",
            },
        ]
        findings = []
        for workflow in workflows:
            present = [name for name in workflow["required"] if name in available_fields]
            missing = [name for name in workflow["required"] if name not in available_fields]
            if not missing:
                status = "Ready"
            elif present:
                status = "Partially ready"
            else:
                status = "Not detected"
            findings.append({**workflow, "present": present, "missing": missing, "status": status})
        return findings

    def _overall_assessment(self, layer_profile: dict, field_summaries: list[dict], correlation_findings: dict) -> dict:
        ready = [item for item in field_summaries if item["readiness"] == "Ready for GeoStats workflows"]
        review = [item for item in field_summaries if item["readiness"].startswith("Review")]
        blocked = [item for item in field_summaries if item["readiness"].startswith("Not ready")]
        risks = []
        if layer_profile["is_geographic"]:
            risks.append("The layer uses a geographic CRS. Reproject before distance-band, K-function, GWR, MGWR, or nearest-neighbor workflows.")
        if layer_profile["feature_count"] < 30:
            risks.append("The layer has a small feature count; global and model statistics may be unstable.")
        if layer_profile.get("empty_geometry_count", 0) > 0:
            risks.append(f"{layer_profile['empty_geometry_count']} feature(s) have empty geometry and may be skipped or distort spatial weights.")
        if layer_profile.get("invalid_geometry_count", 0) > 0:
            risks.append(f"{layer_profile['invalid_geometry_count']} feature(s) have invalid geometry. Repair geometries before contiguity, distance, or local statistics.")
        if blocked:
            risks.append(f"{len(blocked)} audited field(s) are not ready because of insufficient valid data or no variation.")
        if review:
            risks.append(f"{len(review)} audited field(s) should be reviewed before formal interpretation.")
        if correlation_findings["high_pairs"]:
            risks.append(f"{len(correlation_findings['high_pairs'])} high-correlation field pair(s) may create multicollinearity in regression, GWR, MGWR, or GLR workflows.")
        if not risks:
            risks.append("No major automatic readiness warning was triggered.")
        summary = (
            f"{len(ready)} ready field(s), {len(review)} field(s) requiring review, "
            f"{len(blocked)} not-ready field(s), {layer_profile['feature_count']} feature(s)."
        )
        return {"ready": ready, "review": review, "blocked": blocked, "risks": risks, "summary": summary}

    def _push_feedback(self, feedback, layer_profile: dict, field_summaries: list[dict], overall: dict) -> None:
        feedback.pushInfo("Data readiness summary: " + overall["summary"])
        feedback.pushInfo(
            "Geometry diagnostics: "
            f"{layer_profile.get('empty_geometry_count', 0)} empty, "
            f"{layer_profile.get('invalid_geometry_count', 0)} invalid, "
            f"{layer_profile.get('multipart_geometry_count', 0)} multipart feature(s)."
        )
        if layer_profile["is_geographic"]:
            feedback.pushWarning("The input CRS is geographic. Distance-based GeoStats tools should use a suitable projected CRS.")
        for risk in overall["risks"]:
            if risk != "No major automatic readiness warning was triggered.":
                feedback.pushWarning(risk)
        for item in field_summaries:
            if item["readiness"] != "Ready for GeoStats workflows":
                feedback.pushInfo(f"{item['field']}: {item['readiness']}.")

    def _write_html(self, path: str, layer_profile: dict, field_summaries: list[dict], correlation_findings: dict, workflow_findings: list[dict], overall: dict) -> None:
        field_rows = "".join(self._field_row(item) for item in field_summaries)
        role_rows = "".join(self._role_row(item) for item in field_summaries)
        correlation_rows = "".join(self._correlation_row(item) for item in correlation_findings["high_pairs"][:25])
        if not correlation_rows:
            correlation_rows = "<tr><td colspan=\"4\">No audited numeric field pairs exceeded the 0.85 absolute correlation warning threshold.</td></tr>"
        workflow_rows = "".join(self._workflow_row(item) for item in workflow_findings)
        risk_items = "".join(f"<li>{html.escape(risk)}</li>" for risk in overall["risks"])
        next_actions = self._next_actions(overall)
        next_action_items = "".join(f"<li>{html.escape(action)}</li>" for action in next_actions)
        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Data Readiness Audit</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #243142; background: #f5f7fb; margin: 0; padding: 24px; line-height: 1.55; }}
.container {{ max-width: 1180px; margin: 0 auto; background: #fff; border: 1px solid #dbe4ef; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.72rem; color: #162231; }}
h2 {{ margin: 28px 0 12px; font-size: 1.12rem; color: #1f2937; border-left: 4px solid #0f766e; padding-left: 10px; }}
.subtitle {{ color: #5f7187; margin: 0 0 20px; }}
.summary {{ background: #ecfdf5; border-left: 5px solid #0f766e; padding: 15px 17px; margin: 18px 0; }}
.risk {{ background: #fff7ed; border-left: 5px solid #ea580c; padding: 15px 17px; margin: 18px 0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 9px 10px; text-align: left; vertical-align: top; font-size: .84rem; }}
th {{ background: #ccfbf1; color: #115e59; text-transform: uppercase; font-size: .70rem; letter-spacing: .05em; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
.ok {{ color: #047857; font-weight: 700; }}
.review {{ color: #b45309; font-weight: 700; }}
.block {{ color: #be123c; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">
<h1>PlanX GeoStats Lab Data Readiness Audit</h1>
<p class="subtitle">A pre-analysis QA report for spatial statistics, hot spot analysis, local outlier detection, and spatial modeling.</p>
<section class="summary"><strong>Executive summary:</strong> {html.escape(overall["summary"])}</section>
<h2>Layer Profile</h2>
<table>
<tbody>
<tr><td>Feature count</td><td>{layer_profile["feature_count"]}</td></tr>
<tr><td>Field count</td><td>{layer_profile["field_count"]}</td></tr>
<tr><td>Geometry</td><td>{html.escape(layer_profile["geometry"])}</td></tr>
<tr><td>Empty geometries</td><td>{layer_profile.get("empty_geometry_count", 0)}</td></tr>
<tr><td>Invalid geometries</td><td>{layer_profile.get("invalid_geometry_count", 0)} of {layer_profile.get("validity_checked_count", 0)} checked</td></tr>
<tr><td>Multipart features</td><td>{layer_profile.get("multipart_geometry_count", 0)}</td></tr>
<tr><td>CRS</td><td>{html.escape(layer_profile["crs_authid"])} - {html.escape(layer_profile["crs_description"])}</td></tr>
<tr><td>Distance-analysis note</td><td>{html.escape(self._crs_note(layer_profile))}</td></tr>
</tbody>
</table>
<h2>Automatic Risk Review</h2>
<section class="risk"><ul>{risk_items}</ul></section>
<h2>Numeric Field Audit</h2>
<table>
<thead><tr><th>Field</th><th>Valid</th><th>Missing</th><th>Min</th><th>Max</th><th>Mean</th><th>Std. dev.</th><th>Unique</th><th>Readiness</th></tr></thead>
<tbody>{field_rows}</tbody>
</table>
<h2>Analysis Role Suggestions</h2>
<p>These suggestions combine simple field-name clues with numeric readiness checks. Treat them as analyst prompts, not automatic truth: the planning meaning of the indicator still controls the final method choice.</p>
<table>
<thead><tr><th>Field</th><th>Suggested role</th><th>Reason</th><th>Likely tools</th></tr></thead>
<tbody>{role_rows}</tbody>
</table>
<h2>Model Multicollinearity Screen</h2>
<p>High pairwise correlation does not automatically invalidate a model, but it can make coefficient signs, variable importance, and local model surfaces unstable. Review these pairs before OLS, GLR, GWR, MGWR, Spatial Lag, or Spatial Error modeling.</p>
<table>
<thead><tr><th>Field A</th><th>Field B</th><th>Correlation</th><th>Complete records</th></tr></thead>
<tbody>{correlation_rows}</tbody>
</table>
<h2>PlanX Sample Workflow Readiness</h2>
<table>
<thead><tr><th>Workflow</th><th>Status</th><th>Target</th><th>Candidate explanatory fields</th><th>Detected fields</th><th>Missing fields</th><th>Starter sequence</th><th>Planning purpose</th></tr></thead>
<tbody>{workflow_rows}</tbody>
</table>
<h2>Recommended Next Action</h2>
<ul>{next_action_items}</ul>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _write_field_audit_csv(self, path: str, field_summaries: list[dict]) -> None:
        headers = [
            "field",
            "total_features",
            "valid_numeric",
            "missing_or_invalid",
            "missing_pct",
            "minimum",
            "maximum",
            "mean",
            "std",
            "unique_values",
            "readiness",
            "analysis_role",
            "role_reason",
            "suggested_tools",
        ]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for item in field_summaries:
                writer.writerow({
                    "field": item["field"],
                    "total_features": item["total"],
                    "valid_numeric": item["valid"],
                    "missing_or_invalid": item["missing"] + item["non_finite"],
                    "missing_pct": f"{item['missing_pct']:.6f}",
                    "minimum": self._csv_number(item["minimum"]),
                    "maximum": self._csv_number(item["maximum"]),
                    "mean": self._csv_number(item["mean"]),
                    "std": self._csv_number(item["std"]),
                    "unique_values": item["unique"],
                    "readiness": item["readiness"],
                    "analysis_role": item["analysis_role"],
                    "role_reason": item["role_reason"],
                    "suggested_tools": item["suggested_tools"],
                })

    def _write_audit_json(
        self,
        path: str,
        layer_profile: dict,
        field_summaries: list[dict],
        correlation_findings: dict,
        workflow_findings: list[dict],
        overall: dict,
    ) -> None:
        payload = {
            "schema": "planx_geostats_data_readiness_audit",
            "schema_version": "1.0",
            "provider": "PlanX GeoStats Lab",
            "summary": overall["summary"],
            "layer_profile": layer_profile,
            "risks": overall["risks"],
            "field_audit": field_summaries,
            "correlation_findings": correlation_findings,
            "workflow_guidance": workflow_findings,
            "recommended_next_actions": self._next_actions(overall),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._json_ready(payload), handle, ensure_ascii=False, indent=2)

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

    def _field_row(self, item: dict) -> str:
        cls = "ok"
        if item["readiness"].startswith("Review"):
            cls = "review"
        elif item["readiness"].startswith("Not ready"):
            cls = "block"
        return (
            "<tr>"
            f"<td><code>{html.escape(item['field'])}</code></td>"
            f"<td>{item['valid']}</td>"
            f"<td>{item['missing'] + item['non_finite']} ({item['missing_pct']:.1f}%)</td>"
            f"<td>{self._fmt(item['minimum'])}</td>"
            f"<td>{self._fmt(item['maximum'])}</td>"
            f"<td>{self._fmt(item['mean'])}</td>"
            f"<td>{self._fmt(item['std'])}</td>"
            f"<td>{item['unique']}</td>"
            f"<td class=\"{cls}\">{html.escape(item['readiness'])}</td>"
            "</tr>"
        )

    def _correlation_row(self, item: dict) -> str:
        cls = "block" if item["abs_correlation"] >= 0.95 else "review"
        return (
            "<tr>"
            f"<td><code>{html.escape(item['left'])}</code></td>"
            f"<td><code>{html.escape(item['right'])}</code></td>"
            f"<td class=\"{cls}\">{item['correlation']:.3f}</td>"
            f"<td>{item['complete_records']}</td>"
            "</tr>"
        )

    def _role_row(self, item: dict) -> str:
        cls = "ok"
        if item["analysis_role"] in ("Review before analysis", "Count or intensity variable"):
            cls = "review"
        elif item["analysis_role"] == "Do not model yet":
            cls = "block"
        return (
            "<tr>"
            f"<td><code>{html.escape(item['field'])}</code></td>"
            f"<td class=\"{cls}\">{html.escape(item['analysis_role'])}</td>"
            f"<td>{html.escape(item['role_reason'])}</td>"
            f"<td>{html.escape(item['suggested_tools'])}</td>"
            "</tr>"
        )

    def _workflow_row(self, item: dict) -> str:
        cls = "ok" if item["status"] == "Ready" else "review" if item["status"] == "Partially ready" else "block"
        present = ", ".join(item["present"]) or "None"
        missing = ", ".join(item["missing"]) or "None"
        return (
            "<tr>"
            f"<td><strong>{html.escape(item['name'])}</strong></td>"
            f"<td class=\"{cls}\">{html.escape(item['status'])}</td>"
            f"<td>{html.escape(item['target'])}</td>"
            f"<td>{html.escape(item['explanatory'])}</td>"
            f"<td>{html.escape(present)}</td>"
            f"<td>{html.escape(missing)}</td>"
            f"<td><strong>{html.escape(item['tools'])}</strong><br>{html.escape(item['sequence'])}</td>"
            f"<td>{html.escape(item['purpose'])}</td>"
            "</tr>"
        )

    def _next_actions(self, overall: dict) -> list[str]:
        actions = []
        if overall["blocked"]:
            actions.append("Exclude not-ready fields from formal statistics until missing values, field types, or lack of variation are resolved.")
        if overall["review"]:
            actions.append("Inspect fields marked for review in the attribute table and decide whether imputation, filtering, or a more appropriate indicator is needed.")
        actions.append("Before regression or local modeling, avoid putting strongly correlated explanatory variables in the same model unless there is a clear analytical reason.")
        actions.append("Repair invalid geometries and remove or correct empty geometries before using contiguity weights, local statistics, or distance-based models.")
        actions.append("For distance-band, K-function, nearest-neighbor, GWR, MGWR, and spatial regression workflows, verify that the layer CRS uses appropriate projected map units.")
        actions.append("Run Calculate Distance Band or Incremental Autocorrelation before finalizing a distance threshold for global or local spatial statistics.")
        actions.append("Use this report as the first audit artifact in the analysis folder before exporting final HTML reports from the statistical tools.")
        return actions

    def _crs_note(self, layer_profile: dict) -> str:
        if layer_profile["is_geographic"]:
            return "Geographic CRS detected. Reproject before interpreting distances, areas, or neighborhood thresholds."
        return "Projected or non-geographic CRS detected. Still confirm that map units match the planning scale of the analysis."

    def _fmt(self, value) -> str:
        if value is None:
            return "n/a"
        try:
            if not math.isfinite(float(value)):
                return "n/a"
            return f"{float(value):.6g}"
        except Exception:
            return "n/a"

    def _csv_number(self, value) -> str:
        if value is None:
            return ""
        try:
            if not math.isfinite(float(value)):
                return ""
            return f"{float(value):.12g}"
        except Exception:
            return ""
