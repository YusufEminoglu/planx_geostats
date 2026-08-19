# -*- coding: utf-8 -*-
"""Colocation Quotient (CLQ) Processing Algorithm."""
from __future__ import annotations

import os
import tempfile
import html
import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterString,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml,
)

from ..core.weights import geometry_centroid_point
from ..core.advanced_stats_engines import calculate_colocation_quotient
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, histogram_svg

from ._icons import algorithm_icon


class ColocationQuotientAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    CATEGORY_A = "CATEGORY_A"
    CATEGORY_B = "CATEGORY_B"
    KNN = "KNN"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "colocation_quotient"

    def displayName(self) -> str:
        return "Colocation Quotient (CLQ)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("colocation_quotient")

    def createInstance(self):
        return ColocationQuotientAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Tests whether one category of point features (Category B) tends "
            "to be spatially co-located with another category (Category A) "
            "more, less, or exactly as often as chance would predict (Leslie "
            "& Kronenfeld, 2011), using each Category A feature's K nearest "
            "neighbors. This fills a real gap in the plugin: every other "
            "hot-spot/outlier tool here needs a continuous numeric field, "
            "while CLQ is built for categorical point data - land-use type, "
            "business category, incident type.\n\n"
            "CLQ(A to B) = 1 means B appears among A's neighbors exactly as "
            "often as B's overall share of the data; CLQ > 1 means A "
            "attracts B (positive co-location); CLQ < 1 means A repels B "
            "(negative co-location / avoidance). Critically, CLQ is "
            "ASYMMETRIC: CLQ(A to B) is not generally the same as CLQ(B to "
            "A) - this is the statistic's key advantage over symmetric "
            "measures like Ripley's cross-K, and both directions are worth "
            "computing when the relationship isn't obviously one-directional. "
            "Self-colocation (Category A = Category B) is also a valid, "
            "meaningful special case - it tests whether a single category "
            "clusters with itself.\n\n"
            "Uses point centroids (any input geometry type is reduced to its "
            "centroid), so multipart or elongated polygon geometries "
            "simplify complex spatial form the same way every other "
            "centroid-based tool in this plugin does. Significance is "
            "assessed by permutation (999 category-label shuffles by "
            "default, seed 42), since no simple closed-form variance exists "
            "for a K-nearest-neighbor-based colocation measure."
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
                "Category field",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.CATEGORY_A,
                "Category A value (the reference category)",
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.CATEGORY_B,
                "Category B value (the target category to test for co-location with A)",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of nearest neighbors (K)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERMUTATIONS,
                "Number of permutations (Monte Carlo)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=999,
                minValue=99,
                maxValue=9999,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RANDOM_SEED,
                "Random seed (reproducibility)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=42,
                minValue=0,
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
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Colocation Quotient diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        category_a = self.parameterAsString(parameters, self.CATEGORY_A, context)
        category_b = self.parameterAsString(parameters, self.CATEGORY_B, context)
        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "colocation_quotient_report.html")

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Category field '{field_name}' not found.")

        feedback.pushInfo("Extracting point centroids and category values...")
        x_coords, y_coords, cats = [], [], []
        skipped = 0
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val = f.attribute(field_name)
            if val is None or val == NULL or str(val) == "NULL":
                skipped += 1
                continue
            centroid = geometry_centroid_point(f.geometry())
            if centroid is None:
                skipped += 1
                continue
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())
            cats.append(str(val))

        n = len(cats)
        if n <= k_neighbors:
            raise QgsProcessingException(
                f"Only {n} valid feature(s) available; need more than K={k_neighbors} to compute Colocation Quotient."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing category or geometry.")

        cats_arr = np.array(cats)
        if category_a not in cats_arr:
            raise QgsProcessingException(f"Category A value '{category_a}' was not found in field '{field_name}'.")
        if category_b not in cats_arr:
            raise QgsProcessingException(f"Category B value '{category_b}' was not found in field '{field_name}'.")

        feedback.pushInfo(f"Calculating Colocation Quotient using K={k_neighbors} and {permutations} permutations...")
        result = calculate_colocation_quotient(
            cats_arr,
            np.array(x_coords),
            np.array(y_coords),
            category_a,
            category_b,
            k_neighbors=k_neighbors,
            permutations=permutations,
            seed=seed,
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"CLQ({category_a} -> {category_b}) = {result['clq']:.4f}, "
            f"z = {result['z_score']:.3f}, p = {result['p_value']:.4f}"
        )

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(html_path, field_name, category_a, category_b, n, k_neighbors, permutations, result)

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(self, path, field_name, category_a, category_b, n, k_neighbors, permutations, result):
        clq = result["clq"]
        z = result["z_score"]
        p = result["p_value"]
        is_significant = p < 0.05

        if is_significant and clq > 1.0:
            pattern = "Positive Co-Location (Attraction)"
            color = "#e31a1c"
            desc = (
                f"CLQ = {clq:.4f} is significantly above 1.0 (z = {z:.2f}, p = {p:.4f}): "
                f"Category '{html.escape(category_b)}' appears among '{html.escape(category_a)}''s nearest "
                "neighbors more often than its overall prevalence would predict."
            )
        elif is_significant and clq < 1.0:
            pattern = "Negative Co-Location (Avoidance)"
            color = "#1f78b4"
            desc = (
                f"CLQ = {clq:.4f} is significantly below 1.0 (z = {z:.2f}, p = {p:.4f}): "
                f"Category '{html.escape(category_b)}' appears among '{html.escape(category_a)}''s nearest "
                "neighbors less often than its overall prevalence would predict."
            )
        else:
            pattern = "No Significant Co-Location"
            color = "#718096"
            desc = f"CLQ = {clq:.4f} is not distinguishable from 1.0 (p = {p:.4f}): no evidence of co-location or avoidance."

        reverse_note = (
            f"This result is for CLQ('{html.escape(category_a)}' &rarr; '{html.escape(category_b)}') only. "
            f"Re-run with Category A and B swapped to check CLQ('{html.escape(category_b)}' &rarr; '{html.escape(category_a)}') "
            "- the two directions can differ."
        )
        if category_a == category_b:
            reverse_note = f"Self-colocation test: does '{html.escape(category_a)}' cluster with itself more than chance would predict."

        clq_chart = histogram_svg(
            result.get("permuted_values", []),
            observed=clq,
            x_label="Permuted CLQ",
        )

        if is_significant and clq > 1.0:
            next_action = "Map the qualifying Category A features and inspect where the attraction concentrates; consider Local Moran's I or Getis-Ord Gi* on a derived count-density field for a spatial hot-spot view."
        elif is_significant and clq < 1.0:
            next_action = "Investigate whether the avoidance reflects a real repulsion process (e.g. zoning separation) or a sampling/coverage artifact."
        else:
            next_action = "Try a different K (nearest-neighbor count) before concluding no co-location exists - CLQ can be scale-sensitive."

        guidance_html = analyst_guidance_html(
            "Colocation Quotient",
            "The Colocation Quotient tests whether one point category tends to be spatially co-located with another, using each reference-category point's K nearest neighbors, asymmetrically (A to B need not equal B to A).",
            [
                "Point locations (or centroids) are a meaningful representation of the phenomenon being studied.",
                "K is chosen deliberately, not left at a default that doesn't match the expected interaction scale.",
                "Both categories have enough features present for a stable local-CLQ average.",
            ],
            [
                "Treating CLQ(A to B) and CLQ(B to A) as interchangeable - compute both when the direction of attraction/avoidance matters.",
                "Reading CLQ as a measure of overall category prevalence rather than spatial arrangement - a rare category can still have a high CLQ if it's tightly co-located wherever it does occur.",
                "Using a K so large it spans the whole study area, which pulls every local CLQ toward 1.0 regardless of real local structure.",
            ],
            [
                "Join Count Statistics for a polygon-adjacency version of the same categorical clustering question",
                "Local Moran's I / Getis-Ord Gi* on a category-count-density field for a complementary hot-spot view",
                "Average Nearest Neighbor to check the reference category's own overall clustering before interpreting co-location with another category",
            ],
            "Use CLQ when the planning question is genuinely about which categories co-occur spatially (e.g. does affordable housing co-locate with transit access points), not as a general-purpose hot-spot detector - it answers a categorical relationship question, not a magnitude question.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Colocation Quotient Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 780px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    .interpretation-box {{ background-color: #f8fafc; border-left: 5px solid {color}; padding: 20px; border-radius: 4px; margin-bottom: 30px; }}
    .status-title {{ font-size: 1.3rem; font-weight: 800; color: {color}; margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.05em; }}
    .status-desc {{ color: #4a5568; font-size: 0.95rem; margin: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 0.9rem; }}
    th {{ background-color: #ebf8ff; color: #2b6cb0; font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    .metric-name {{ font-weight: 600; color: #2d3748; }}
    .metric-val {{ font-family: monospace; font-size: 1rem; font-weight: 600; }}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
    section {{ margin: 28px 0; }}
    h2 {{ color: #1a202c; font-size: 1.15rem; margin: 0 0 12px 0; }}
    .next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
    .note-box {{ background: #fffaf0; border-left: 5px solid #dd6b20; padding: 14px 16px; border-radius: 4px; font-size: .88rem; }}
    {analyst_guidance_css()}
    {chart_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Colocation Quotient (CLQ)</h1>
        <p class="subtitle">Field: <strong>{html.escape(field_name)}</strong> | A &rarr; B: <strong>{html.escape(category_a)} &rarr; {html.escape(category_b)}</strong> | N: <strong>{n}</strong> | K: <strong>{k_neighbors}</strong> | Permutations: <strong>{permutations}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">{pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>The Colocation Quotient (Leslie &amp; Kronenfeld, 2011) measures whether Category B is over- or under-represented among Category A's K nearest neighbors, relative to B's overall prevalence. CLQ = 1 is the no-colocation reference point; significance is assessed via {permutations} random category-label permutations.</p>
    </section>

    <table>
        <thead><tr><th>Colocation Quotient Diagnostic</th><th>Statistical Value</th></tr></thead>
        <tbody>
            <tr><td class="metric-name">CLQ(A &rarr; B)</td><td class="metric-val">{clq:.6f}</td></tr>
            <tr><td class="metric-name">Category A count (N_A)</td><td class="metric-val">{result['n_a']}</td></tr>
            <tr><td class="metric-name">Category B count (N_B)</td><td class="metric-val">{result['n_b']}</td></tr>
            <tr><td class="metric-name">Total valid features (N)</td><td class="metric-val">{result['n_total']}</td></tr>
            <tr><td class="metric-name">Permuted Mean</td><td class="metric-val">{result['permuted_mean']:.6f}</td></tr>
            <tr><td class="metric-name">Permuted Std. Dev.</td><td class="metric-val">{result['permuted_std']:.6f}</td></tr>
            <tr><td class="metric-name">z-score</td><td class="metric-val">{z:.6f}</td></tr>
            <tr><td class="metric-name">p-value</td><td class="metric-val">{p:.6f}</td></tr>
        </tbody>
    </table>

    <div class="note-box">{reverse_note}</div>

    <section>
        <h2>Permutation Reference Distribution</h2>
        {clq_chart}
        <p class="chart-caption">Distribution of CLQ(A &rarr; B) computed on {permutations} random reassignments of category labels across the same point locations. The dashed line marks the observed CLQ.</p>
    </section>

    <section style="margin-top:28px;">
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>

    {guidance_html}

    <footer>
        Generated by PlanX GeoStats Lab global spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)
