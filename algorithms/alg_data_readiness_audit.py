# -*- coding: utf-8 -*-
"""Data readiness audit for PlanX GeoStats Lab workflows."""
from __future__ import annotations

import html
import math
import os
import tempfile
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
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Data readiness audit report"))
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

        feedback.pushInfo(f"Auditing {len(audit_fields)} numeric field(s) for GeoStats readiness.")
        layer_profile = self._layer_profile(source)
        layer_profile.update(self._geometry_diagnostics(source, feedback))
        field_summaries, field_values = self._field_summaries(source, audit_fields, feedback)
        correlation_findings = self._correlation_findings(field_values)
        workflow_findings = self._workflow_findings({item["field"] for item in field_summaries})
        overall = self._overall_assessment(layer_profile, field_summaries, correlation_findings)

        self._push_feedback(feedback, layer_profile, field_summaries, overall)
        self._write_html(html_path, layer_profile, field_summaries, correlation_findings, workflow_findings, overall)

        return {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
            self.SUMMARY: overall["summary"],
        }

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
            })
        return summaries, aligned_values

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
                "tools": "Global Moran, Incremental Autocorrelation, Gi*, Local Moran",
                "purpose": "Find whether heat-related indicators cluster, disperse, or form local hot and cold spots.",
            },
            {
                "name": "Green cooling and built-form model",
                "required": ["median_land_surface_temp_c", "median_ndvi", "park_m2_per_capita", "impervious_surface_pct", "building_coverage_pct"],
                "tools": "OLS, GLR, GWR, MGWR, Spatial Lag, Spatial Error, Model Comparison Matrix",
                "purpose": "Explain temperature variation using vegetation, public green access, imperviousness, and urban form.",
            },
            {
                "name": "Accessibility and network structure",
                "required": ["street_connectivity", "normalized_integration"],
                "tools": "Similarity Search, Multivariate Clustering, regression diagnostics",
                "purpose": "Compare neighborhoods by movement-network indicators and identify similar planning profiles.",
            },
            {
                "name": "Equity and vulnerable population review",
                "required": ["senior_65plus_population", "youth_population", "park_m2_per_capita"],
                "tools": "Gi*, Local Moran, Bivariate Lee's L, Exploratory Regression",
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
<h2>Model Multicollinearity Screen</h2>
<p>High pairwise correlation does not automatically invalidate a model, but it can make coefficient signs, variable importance, and local model surfaces unstable. Review these pairs before OLS, GLR, GWR, MGWR, Spatial Lag, or Spatial Error modeling.</p>
<table>
<thead><tr><th>Field A</th><th>Field B</th><th>Correlation</th><th>Complete records</th></tr></thead>
<tbody>{correlation_rows}</tbody>
</table>
<h2>PlanX Sample Workflow Readiness</h2>
<table>
<thead><tr><th>Workflow</th><th>Status</th><th>Detected fields</th><th>Missing fields</th><th>Recommended tools</th><th>Planning purpose</th></tr></thead>
<tbody>{workflow_rows}</tbody>
</table>
<h2>Recommended Next Action</h2>
<ul>{next_action_items}</ul>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

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

    def _workflow_row(self, item: dict) -> str:
        cls = "ok" if item["status"] == "Ready" else "review" if item["status"] == "Partially ready" else "block"
        present = ", ".join(item["present"]) or "None"
        missing = ", ".join(item["missing"]) or "None"
        return (
            "<tr>"
            f"<td><strong>{html.escape(item['name'])}</strong></td>"
            f"<td class=\"{cls}\">{html.escape(item['status'])}</td>"
            f"<td>{html.escape(present)}</td>"
            f"<td>{html.escape(missing)}</td>"
            f"<td>{html.escape(item['tools'])}</td>"
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
