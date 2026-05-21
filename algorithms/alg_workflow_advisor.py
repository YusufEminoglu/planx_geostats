# -*- coding: utf-8 -*-
"""Workflow advisor for PlanX GeoStats Lab tools."""
from __future__ import annotations

import html
import os
import tempfile

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFileDestination,
)

from ._icons import algorithm_icon


class GeoStatsWorkflowAdvisorAlgorithm(QgsProcessingAlgorithm):
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "geostats_workflow_advisor"

    def displayName(self) -> str:
        return "GeoStats Workflow Advisor"

    def group(self) -> str:
        return "00 | Setup and Diagnostics"

    def groupId(self) -> str:
        return "planx_setup_diagnostics"

    def icon(self):
        return algorithm_icon("geostats_workflow_advisor")

    def createInstance(self):
        return GeoStatsWorkflowAdvisorAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Creates a practical HTML advisor that maps common planning-analysis "
            "questions to PlanX GeoStats Lab tools, input requirements, expected "
            "outputs, and follow-up checks.\n\n"
            "Use this before choosing a statistic, when teaching a workflow, or when "
            "documenting which tool sequence should be used for a planning question."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output workflow advisor",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "GeoStats workflow advisor"))

    def processAlgorithm(self, parameters, context, feedback):
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_geostats_workflow_advisor.html")

        feedback.pushInfo("Writing PlanX GeoStats workflow advisor...")
        self._write_html(html_path)
        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path: str) -> None:
        starter_rows = [
            (
                "Is my variable spatially clustered across the study area?",
                "Global Moran's I; Incremental Spatial Autocorrelation",
                "One numeric field on point or polygon features; projected CRS for distance bands.",
                "Global z-score, p-value, Moran's I, and a defensible distance scale.",
                "If significant, continue to Local Moran's I or Getis-Ord Gi* to map where the pattern occurs.",
            ),
            (
                "Where are statistically strong hot and cold spots?",
                "Getis-Ord Gi*; Local Moran's I",
                "Numeric field and a neighborhood rule that avoids too many isolated or fully connected features.",
                "Feature layer with hot/cold confidence classes or cluster/outlier classes.",
                "Compare hot spots with planning constraints, vulnerable population, green-space access, or infrastructure layers.",
            ),
            (
                "What distance band should I use?",
                "Calculate Distance Band; Incremental Spatial Autocorrelation",
                "Projected coordinates and enough features to support neighbor search.",
                "Suggested neighbor distance and peak spatial autocorrelation distance.",
                "Use the selected distance in Moran, Gi*, Local Moran, General G, or sensitivity testing.",
            ),
            (
                "Is the whole pattern clustered, dispersed, or random?",
                "Average Nearest Neighbor; Ripley's K; General G",
                "Point data for ANN/Ripley's K; numeric intensity field for General G.",
                "Observed-vs-expected distance or K curves, z-score, and pattern interpretation.",
                "Use results as a screening step before local statistics or regression modeling.",
            ),
            (
                "Where is the center, spread, or directional trend?",
                "Mean Center; Median Center; Central Feature; Standard Distance; Directional Distribution; Linear Directional Mean",
                "Point/polygon centroids for center/spread tools; line features for directional mean.",
                "Representative center, central feature, standard distance circle, ellipse, or trend line.",
                "Use results for descriptive planning summaries and before/after scenario comparisons.",
            ),
            (
                "Which explanatory variables best explain an outcome?",
                "OLS Regression; Exploratory Regression; Generalized Linear Regression",
                "Complete numeric dependent and explanatory fields; binary target for logistic GLR; counts for Poisson GLR.",
                "Coefficients, model fit, residuals, multicollinearity warnings, and diagnostic report.",
                "Map residuals and check residual spatial autocorrelation before trusting coefficient narratives.",
            ),
            (
                "Do relationships vary across space?",
                "GWR; MGWR",
                "Projected coordinates, enough complete records, and non-collinear predictors.",
                "Local coefficients, local fit, residuals, bandwidth information, and diagnostic report.",
                "Compare local coefficient surfaces with planning zones and known spatial processes.",
            ),
            (
                "Does spatial dependence remain in my model?",
                "Spatial Lag Regression; Spatial Error Regression; Model Comparison Matrix",
                "Optional PySAL/spreg libraries, a neighborhood rule, and complete model fields.",
                "Spatial rho/lambda, predicted values, residuals, and model comparison metrics.",
                "Use model comparison to decide whether OLS, GLR, GWR, SAR, SEM, or MGWR is more defensible.",
            ),
        ]
        qa_rows = [
            ("Before analysis", "Data Readiness Audit", "Check CRS, geometry validity, missing values, constant fields, outliers, and correlation risk."),
            ("Before distance tools", "Calculate Distance Band", "Confirm projected units and avoid arbitrary thresholds."),
            ("Before advanced models", "GeoStats Library Status", "Confirm optional libraries are installed in the active QGIS Python environment."),
            ("Before release/manual QA", "Sample Dataset Guide", "Load the Izmir planning sample, synthetic QA fixture, or both."),
            ("After modeling", "Model Comparison Matrix", "Compare fit, residual, and prediction fields across multiple model outputs."),
        ]
        selection_rows = [
            ("Point pattern without attributes", "Average Nearest Neighbor; Ripley's K", "Use ANN for one-distance summary; use Ripley's K to inspect pattern across multiple distances."),
            ("Numeric value over polygons or points", "Global Moran's I; General G; Incremental Spatial Autocorrelation", "Use global tools to confirm whether a variable has spatial structure before mapping local clusters."),
            ("Local clusters and outliers", "Getis-Ord Gi*; Local Moran's I; Bivariate Lee's L", "Use Gi* for hot/cold intensity, Local Moran for high-high/low-low/outlier classes, and Lee's L for paired-variable association."),
            ("Descriptive geography", "Mean Center; Median Center; Central Feature; Standard Distance; Directional Distribution; Linear Directional Mean", "Use these for concise spatial summaries, scenario comparisons, and directional trend communication."),
            ("Explanatory model", "OLS; GLR; Exploratory Regression", "Use OLS for continuous outcomes, logistic GLR for binary outcomes, Poisson GLR for counts, and Exploratory Regression for candidate screening."),
            ("Spatially varying model", "GWR; MGWR", "Use when a global model hides local variation and the sample size supports local estimation."),
            ("Spatial dependence model", "Spatial Lag Regression; Spatial Error Regression", "Use when residual diagnostics or theory suggest spatial spillovers or spatially structured omitted variables."),
            ("Model audit", "Model Comparison Matrix; Sensitivity Test", "Use comparison and randomization checks before presenting a preferred planning model."),
        ]
        assumption_rows = [
            ("Distance-based tools", "Projected CRS, meaningful map units, and a defensible neighborhood scale.", "Geographic degrees, arbitrary thresholds, or many isolated observations."),
            ("Local cluster tools", "Enough neighbors per feature and a field with real variation.", "Sparse support, all-connected graphs, multiple testing, and boundary effects."),
            ("OLS / GLR", "Complete records, non-constant predictors, and a model family that matches the outcome.", "Multicollinearity, misspecified family, outliers, and residual spatial pattern."),
            ("GWR / MGWR", "A sample large enough for local estimation and predictors that vary locally.", "Overfitting, unstable local coefficients, bandwidth misuse, and strong collinearity."),
            ("Spatial Lag / Error", "A theory-driven spatial weights model and optional PySAL/spreg availability.", "Treating spatial dependence as proof of causality or ignoring islands in the weights graph."),
        ]
        pitfall_rows = [
            ("Using WGS84 distances", "Distances are measured in degrees, not meters.", "Reproject to a suitable local projected CRS before distance bands, ANN, Ripley's K, GWR, SAR, or SEM."),
            ("Choosing one arbitrary threshold", "Cluster conclusions can flip when the distance band changes.", "Run Calculate Distance Band, Incremental Autocorrelation, and Sensitivity Test."),
            ("Skipping data readiness", "Nulls, constants, and geometry problems can silently remove records.", "Run Data Readiness Audit and review skipped-record counts in every report."),
            ("Reading p-values alone", "A significant result can still be practically weak or methodologically fragile.", "Check maps, effect sizes, neighborhood diagnostics, and planning context together."),
            ("Comparing models by one metric", "AIC, R2, residual maps, and spatial diagnostics can disagree.", "Use Model Comparison Matrix and inspect residual spatial autocorrelation."),
        ]
        recipe_rows = [
            ("Urban heat clustering", "Izmir sample", "median_land_surface_temp_c; median_heat_island_index", "Data Readiness Audit -> Incremental Autocorrelation -> Global Moran's I -> Getis-Ord Gi*"),
            ("Green cooling relationship", "Izmir sample", "median_land_surface_temp_c with median_ndvi, park_m2_per_capita, tree_canopy_coverage_pct", "OLS -> GWR -> residual spatial autocorrelation -> Model Comparison Matrix"),
            ("Vulnerable population concentration", "Izmir sample", "senior_65plus_population; child_population", "Global Moran's I -> Local Moran's I -> Getis-Ord Gi*"),
            ("Count model smoke test", "Synthetic QA fixture", "count_target with explanatory_a and explanatory_b", "Generalized Linear Regression with Poisson family"),
            ("Directional line QA", "Synthetic QA fixture", "qa_lines_directional", "Linear Directional Mean"),
        ]

        starter_body = "".join(self._workflow_row(*row) for row in starter_rows)
        qa_body = "".join(self._three_column_row(stage, tool, check, emphasize_second=True) for stage, tool, check in qa_rows)
        selection_body = "".join(self._three_column_row(data_situation, tools, decision_rule, emphasize_first=True) for data_situation, tools, decision_rule in selection_rows)
        assumption_body = "".join(self._three_column_row(family, assumption, caution, emphasize_first=True) for family, assumption, caution in assumption_rows)
        pitfall_body = "".join(self._three_column_row(pitfall, risk, safer_move, emphasize_first=True) for pitfall, risk, safer_move in pitfall_rows)
        recipe_body = "".join(
            "<tr>"
            f"<td><strong>{html.escape(question)}</strong></td>"
            f"<td>{html.escape(dataset)}</td>"
            f"<td><code>{html.escape(fields)}</code></td>"
            f"<td>{html.escape(sequence)}</td>"
            "</tr>"
            for question, dataset, fields, sequence in recipe_rows
        )
        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Workflow Advisor</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #263241; background: #f6f8fb; margin: 0; padding: 24px; line-height: 1.55; }}
.container {{ max-width: 1180px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.72rem; color: #17212f; }}
h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; border-left: 4px solid #1d4ed8; padding-left: 10px; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.note {{ background: #eff6ff; border-left: 5px solid #2563eb; padding: 14px 18px; margin: 20px 0; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .88rem; }}
th {{ background: #dbeafe; color: #1e3a8a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.question {{ font-weight: 700; color: #17212f; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="container">
<h1>GeoStats Workflow Advisor</h1>
<p class="subtitle">A planning-oriented guide for choosing PlanX GeoStats Lab tools and sequencing common spatial-statistics workflows.</p>
<div class="note"><strong>Recommended first step:</strong> run <strong>Data Readiness Audit</strong> on the candidate layer, then choose a workflow below. Distance-based tools should generally use a suitable projected CRS.</div>
<h2>Planning Questions to Tool Sequences</h2>
<table>
<thead><tr><th>Planning question</th><th>Recommended tools</th><th>Input requirements</th><th>Expected output</th><th>Next step</th></tr></thead>
<tbody>{starter_body}</tbody>
</table>
<h2>Tool Selection Matrix</h2>
<table>
<thead><tr><th>Data or question type</th><th>Candidate tools</th><th>Selection rule</th></tr></thead>
<tbody>{selection_body}</tbody>
</table>
<h2>Method Assumptions and Cautions</h2>
<table>
<thead><tr><th>Method family</th><th>Minimum assumptions</th><th>Main cautions</th></tr></thead>
<tbody>{assumption_body}</tbody>
</table>
<h2>Common Pitfalls and Safer Moves</h2>
<table>
<thead><tr><th>Pitfall</th><th>Why it matters</th><th>Safer move</th></tr></thead>
<tbody>{pitfall_body}</tbody>
</table>
<h2>Starter Recipes for Bundled Samples</h2>
<table>
<thead><tr><th>Question</th><th>Dataset</th><th>Starter fields or layer</th><th>Suggested sequence</th></tr></thead>
<tbody>{recipe_body}</tbody>
</table>
<h2>Quality Gates</h2>
<table>
<thead><tr><th>Stage</th><th>Tool</th><th>Why it matters</th></tr></thead>
<tbody>{qa_body}</tbody>
</table>
<h2>Interpretation Discipline</h2>
<ul>
<li>Do not interpret statistical significance without checking neighborhood support, CRS units, missing records, and field distribution.</li>
<li>Use global tools to screen pattern, local tools to locate pattern, and models to test explanatory relationships.</li>
<li>Prefer comparing several defensible neighborhood definitions over relying on one arbitrary distance threshold.</li>
<li>For planning decisions, combine statistical outputs with policy constraints, local knowledge, and map review.</li>
<li>Before release or classroom use, run the manual QA scenarios documented in <code>QA_MANUAL_TEST_MATRIX.md</code>.</li>
</ul>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def _workflow_row(self, question: str, tools: str, requirements: str, output: str, next_step: str) -> str:
        return (
            "<tr>"
            f"<td class=\"question\">{html.escape(question)}</td>"
            f"<td>{html.escape(tools)}</td>"
            f"<td>{html.escape(requirements)}</td>"
            f"<td>{html.escape(output)}</td>"
            f"<td>{html.escape(next_step)}</td>"
            "</tr>"
        )

    def _three_column_row(self, first: str, second: str, third: str, emphasize_first: bool = False, emphasize_second: bool = False) -> str:
        first_value = f"<strong>{html.escape(first)}</strong>" if emphasize_first else html.escape(first)
        second_value = f"<strong>{html.escape(second)}</strong>" if emphasize_second else html.escape(second)
        return (
            "<tr>"
            f"<td>{first_value}</td>"
            f"<td>{second_value}</td>"
            f"<td>{html.escape(third)}</td>"
            "</tr>"
        )
