# -*- coding: utf-8 -*-
"""Join Count Statistics (BB/WW/BW) Processing Algorithm."""
from __future__ import annotations

import os
import tempfile
import html

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFileDestination,
    QgsProcessingOutputHtml,
)

from ..core.weights import build_weights_matrix
from ..core.advanced_stats_engines import calculate_join_counts
from ..core.analysis_diagnostics import (
    crs_unit_warning,
    neighbor_summary,
    numeric_quality_summary,
    push_diagnostics,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html

from ._icons import algorithm_icon


class JoinCountStatisticsAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    CATEGORY_VALUE = "CATEGORY_VALUE"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "join_count_statistics"

    def displayName(self) -> str:
        return "Join Count Statistics (BB/WW/BW)"

    def group(self) -> str:
        return "02 | Urban Pattern Scan"

    def groupId(self) -> str:
        return "planx_pattern_scan"

    def icon(self):
        return algorithm_icon("join_count_statistics")

    def createInstance(self):
        return JoinCountStatisticsAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Tests spatial clustering of a binary/categorical variable by "
            "counting how often neighboring feature pairs (joins) share the "
            "same category versus differ (Cliff & Ord, 1973, 1981). Every "
            "other autocorrelation tool in this plugin needs a continuous "
            "numeric field - Join Count Statistics is the one built for "
            "presence/absence, zoning class, or any yes/no coding.\n\n"
            "The target field is converted to binary using the category "
            "value you supply: features matching it become 'Black' (1), "
            "everything else becomes 'White' (0). Three join counts are "
            "reported, each with its own permutation-based significance "
            "test (999 shuffles by default, seed 42):\n"
            "- BB (Black-Black): neighboring pairs that both match the "
            "category - high BB means the category clusters together.\n"
            "- WW (White-White): neighboring pairs that both do NOT match - "
            "high WW means the complement clusters together.\n"
            "- BW (Black-White): neighboring pairs that differ - high BW "
            "means the category is spatially interspersed/checkerboarded "
            "rather than clustered.\n\n"
            "Unlike every weights-based tool in this plugin, Join Counts "
            "use the raw (unstandardized) adjacency graph, not row-"
            "standardized weights - the count of actual shared edges is "
            "what the test statistic needs, not a per-feature average. "
            "Typical uses: does a land-use category (e.g. 'arterial "
            "corridor', 'gridiron_index above a threshold') form contiguous "
            "clusters, or is it scattered piecemeal across the study area?"
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
                "Field to convert to binary category",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.CATEGORY_VALUE,
                "Reserved (unused) - category value entered below",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any,
                optional=True,
                defaultValue=None,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / adjacency type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
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
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "Join Count Statistics diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "join_count_report.html")

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")
        field_type = source.fields().field(field_idx).type()
        is_numeric_field = field_type in (2, 4, 6)  # Int, LongLong, Double

        feedback.pushInfo("Generating spatial adjacency graph...")
        neighbors, _weights, id_order, _ = build_weights_matrix(
            source, weight_type, k_neighbors=k_neighbors, distance_band=distance_band, feedback=feedback
        )
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Reading and binarizing target field...")
        raw_values = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val = f.attribute(field_name)
            if val is None or val == NULL or str(val) == "NULL":
                continue
            raw_values[f.id()] = val

        valid_id_order = [fid for fid in id_order if fid in raw_values]
        if len(valid_id_order) <= 3:
            raise QgsProcessingException("At least 4 valid features are required for Join Count analysis.")

        if is_numeric_field:
            numeric_vals = [float(raw_values[fid]) for fid in valid_id_order]
            threshold = sorted(numeric_vals)[len(numeric_vals) // 2]
            x_binary = {fid: (1 if float(raw_values[fid]) > threshold else 0) for fid in valid_id_order}
            category_desc = f"values above the median ({threshold:.4f})"
        else:
            values_present = [raw_values[fid] for fid in valid_id_order]
            most_common = max(set(str(v) for v in values_present), key=lambda v: sum(1 for x in values_present if str(x) == v))
            x_binary = {fid: (1 if str(raw_values[fid]) == most_common else 0) for fid in valid_id_order}
            category_desc = f"category = '{most_common}' (most frequent value)"

        neighborhood_summary = neighbor_summary(neighbors, valid_id_order)
        crs_warning = crs_unit_warning(source)
        numeric_summary = numeric_quality_summary(source.featureCount(), x_binary, list(x_binary.values()))
        push_diagnostics(feedback, numeric_summary, neighborhood_summary, crs_warning)

        if len(set(x_binary.values())) < 2:
            raise QgsProcessingException("Join Count Statistics requires both categories (Black and White) to be present after binarization.")

        feedback.pushInfo(f"Calculating Join Counts using {permutations} permutations...")
        x_array = [x_binary[fid] for fid in valid_id_order]
        result = calculate_join_counts(x_array, neighbors, valid_id_order, permutations=permutations, seed=seed)

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"BB={result['bb']['observed']} (p={result['bb']['p_value']:.4f}), "
            f"WW={result['ww']['observed']} (p={result['ww']['p_value']:.4f}), "
            f"BW={result['bw']['observed']} (p={result['bw']['p_value']:.4f})"
        )

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, category_desc, len(valid_id_order), result,
            neighborhood_summary, crs_warning, weight_type, permutations,
        )

        return {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def write_html_report(
        self, path, field_name, category_desc, n, result, neighborhood_summary, crs_warning, weight_type, permutations,
    ):
        bb, ww, bw = result["bb"], result["ww"], result["bw"]

        clustering_signals = []
        if bb["p_value"] < 0.05 and bb["observed"] > bb["permuted_mean"]:
            clustering_signals.append("the category clusters together (BB significant)")
        if ww["p_value"] < 0.05 and ww["observed"] > ww["permuted_mean"]:
            clustering_signals.append("the complement clusters together (WW significant)")
        if bw["p_value"] < 0.05 and bw["observed"] > bw["permuted_mean"]:
            clustering_signals.append("the category is spatially interspersed / checkerboarded (BW significant)")

        if clustering_signals:
            pattern = "Significant Spatial Pattern Detected"
            color = "#e31a1c"
            desc = "; ".join(clustering_signals).capitalize() + "."
        else:
            pattern = "Random (No Significant Join Pattern)"
            color = "#718096"
            desc = "None of BB, WW, or BW join counts differ significantly from a random spatial arrangement."

        if neighborhood_summary["isolated"] > 0:
            next_action = "Increase the neighborhood size before interpreting join counts; isolated features contribute zero joins."
        elif bb["p_value"] < 0.05 and bb["observed"] > bb["permuted_mean"]:
            next_action = "Delineate the clustered zone with Local Moran's I or Getis-Ord Gi* on a continuous proxy field, or map the matching features directly."
        elif bw["p_value"] < 0.05 and bw["observed"] > bw["permuted_mean"]:
            next_action = "Investigate whether the interspersion reflects a genuine fine-grained mixed pattern or a data-coding artifact (e.g. alternating survey/parcel IDs)."
        else:
            next_action = "No further spatial follow-up needed for this category coding; consider a different threshold or field if clustering is expected but not detected."

        rows_html = ""
        for label, sub in (("BB (Black-Black)", bb), ("WW (White-White)", ww), ("BW (Black-White)", bw)):
            rows_html += (
                f"<tr><td class='metric-name'>{label}</td>"
                f"<td class='metric-val'>{sub['observed']}</td>"
                f"<td class='metric-val'>{sub['permuted_mean']:.2f}</td>"
                f"<td class='metric-val'>{sub['permuted_std']:.3f}</td>"
                f"<td class='metric-val'>{sub['z_score']:.3f}</td>"
                f"<td class='metric-val'>{sub['p_value']:.4f}</td></tr>\n"
            )

        guidance_html = analyst_guidance_html(
            "Join Count Statistics",
            "Join Count Statistics test spatial clustering of a binary/categorical variable by counting same-category (BB), complement (WW), and cross-category (BW) neighbor pairs.",
            [
                "The binarization threshold or category value is meaningful for the analysis question, not an arbitrary median split.",
                "Both categories have enough features present for BB and WW to be statistically interpretable.",
                "The neighborhood graph is not dominated by isolated features.",
            ],
            [
                "Binarizing a continuous field at the median when a natural, meaningful threshold exists (e.g. a regulatory cutoff) - always prefer the meaningful one.",
                "Reading a non-significant BW as proof of clustering - BW near its expected value is uninformative, not confirmatory.",
                "Confusing 'the category clusters' (BB high) with 'the category is common' - Join Counts test spatial arrangement, not prevalence.",
            ],
            [
                "Local Moran's I / Getis-Ord Gi* for continuous fields",
                "Colocation Quotient for multi-category point-pattern co-location",
                "Data Readiness Audit to choose a defensible binarization threshold",
            ],
            "Use Join Count Statistics when the question is genuinely about a category or presence/absence coding, not as a workaround for skipping continuous-field autocorrelation tools by binarizing unnecessarily.",
        )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Join Count Statistics Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 800px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
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
    {analyst_guidance_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Join Count Statistics (BB/WW/BW)</h1>
        <p class="subtitle">Field: <strong>{html.escape(field_name)}</strong> | Category: <strong>{html.escape(category_desc)}</strong> | N: <strong>{n}</strong> | Adjacency: <strong>{html.escape(weight_type)}</strong> | Permutations: <strong>{permutations}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">{pattern}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>Join Count Statistics count same-category (Black-Black) and cross-category (Black-White) neighboring pairs to test spatial clustering of a binary variable, using the raw adjacency graph and {permutations} random permutations for significance.</p>
    </section>

    <table>
        <thead><tr><th>Join Type</th><th>Observed</th><th>Permuted Mean</th><th>Permuted Std.</th><th>z-score</th><th>p-value</th></tr></thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <section>
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
