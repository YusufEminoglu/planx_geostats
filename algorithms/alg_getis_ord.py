# -*- coding: utf-8 -*-
"""Getis-Ord Gi* Hotspot Analysis Processing Algorithm."""
from __future__ import annotations

import html
import logging
from collections import Counter
import os
import tempfile
import numpy as np

from qgis.PyQt.QtCore import QVariant
from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsFeature,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFileDestination,
    QgsFeatureSink
)

from ..core.weights import build_weights_matrix
from ..core.stats_engines import calculate_getis_ord
from ..core.layer_metadata import apply_output_metadata
from ..core.local_pattern_audit import (
    MULTIPLE_TESTING_OPTIONS,
    NO_DATA,
    NO_NEIGHBOURS,
    NOT_STABLE,
    STABLE_COLD,
    STABLE_HOT,
    adjust_p_values,
    correction_words,
    corrected_gi_bins,
    getis_ord_class_summary,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, donut_chart_svg, kpi_card_row_html
from ..core.symbology import (
    GI_CLASS_STYLE,
    GI_CONFIDENCE_STYLE,
    GI_CONSENSUS_STYLE,
    MISSING_CLASS_STYLE,
    apply_renderer,
    gi_class_renderer,
)

from ._icons import algorithm_icon
from ._deliver import style_loaded_output
from ._guards import (
    check_weight_choice,
    engine_errors,
    neighbour_counts,
    push_skips,
    read_finite_values,
    require_neighbours,
    restrict_weights,
    set_fields_keeping_values,
    skip_message,
    stop_if_cancelled,
)

GI_LABEL_BY_BIN = {value: label for value, _color, label in GI_CONFIDENCE_STYLE}


def _variant_label(weight_type, k_neighbors, distance_band):
    if weight_type == "knn":
        return f"'K-nearest neighbours, k = {int(k_neighbors)}'"
    if weight_type == "distance":
        return f"'distance band {distance_band:g}'"
    return f"'{weight_type} contiguity'"


def stability_variants(weight_type, k_neighbors, distance_band):
    """Neighbourhood definitions for the stability consensus.

    Only the chosen family is varied: a distance band is scaled only when a
    distance band was chosen (3.12 always added 0.75x and 1.25x bands, in
    degrees on a geographic CRS and at the 1000 default under KNN).
    """
    if weight_type == "distance":
        return [("distance", k_neighbors, distance_band * f) for f in (0.75, 0.9, 1.1, 1.25)]
    if weight_type == "knn":
        ks = sorted({k for k in (k_neighbors - 2, k_neighbors - 1, k_neighbors + 1, k_neighbors + 2)
                     if k >= 1 and k != k_neighbors})
        return [("knn", k, distance_band) for k in ks]
    other = "rook" if weight_type == "queen" else "queen"
    return [(other, k_neighbors, distance_band),
            ("knn", max(1, k_neighbors - 2), distance_band),
            ("knn", k_neighbors + 2, distance_band)]


logger = logging.getLogger("PlanX GeoStats Lab")


class GetisOrdAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"
    STABILITY_ANALYSIS = "STABILITY_ANALYSIS"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None
        self.renderer_field = "gi_conf"
        self.consensus_map = False

    def name(self) -> str:
        return "getis_ord_gi"

    def displayName(self) -> str:
        return "Hot Spot Analysis (Getis-Ord Gi*)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("getis_ord_gi")

    def createInstance(self):
        return GetisOrdAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes the Getis-Ord Gi* statistic for every feature: whether a "
            "feature and its neighbors together form a statistically significant "
            "hot spot (high values), cold spot (low values), or neither, relative "
            "to the study area as a whole. Gi* always includes the feature itself "
            "in its own neighborhood sum (unlike plain Gi), which behaves better "
            "at the edges of the study area and is the standard variant used in "
            "practice.\n\n"
            "Output fields: gi_zscore, gi_pvalue (raw), gi_padj (after the chosen "
            "multiple-testing correction), gi_conf (a signed confidence bin from "
            "gi_padj: +/-3 at p < 0.01, +/-2 at p < 0.05, +/-1 at p < 0.10, 0 "
            "otherwise; sign gives hot vs cold), gi_class (the class drawn on the "
            "map, in words) and gi_nbrs (how many valid neighbors supported the "
            "local statistic). Regardless of the chosen "
            "weight type (Queen, Rook, KNN, Distance Band), every neighbor "
            "contributes equally - the weight type only decides who counts as a "
            "neighbor, not how strongly; there is no distance-decay weighting "
            "inside Gi* itself.\n\n"
            "A feature with no neighbor that has data gets no test: gi_zscore, "
            "gi_pvalue and gi_padj are NULL, gi_conf = 0 and gi_class = 'No "
            "neighbours'. A feature whose value is NULL, non-numeric, NaN or "
            "infinite is left out (it is nobody's neighbor) and written as gi_conf "
            "= 0, gi_class = 'No data'. Both are drawn in grey and counted in the "
            "report. Queen and Rook are refused on point layers, and a Distance "
            "Band on a geographic CRS (degrees).\n\n"
            "The optional stability consensus re-runs Gi* under neighbouring "
            "definitions of the same family (K +/- 1, 2; distance band x 0.75 to "
            "1.25; the other contiguity plus two KNN runs) and maps gi_cons: "
            "'Stable hot spot' / 'Stable cold spot' when at least 60 % of the "
            "runs agree on the direction and at least three are significant, "
            "otherwise 'Not stable'. That legend is a vote, not a confidence "
            "level. Gi* finds clusters, not outliers - where a high "
            "value surrounded by low neighbors (or vice versa) is the object of "
            "interest, use Local Moran's I instead."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD,
                "Target numeric field to analyze",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MULTIPLE_TESTING,
                "Multiple-testing correction",
                options=[label for label, _ in MULTIPLE_TESTING_OPTIONS],
                defaultValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.STABILITY_ANALYSIS,
                "Run multi-neighborhood stability consensus",
                defaultValue=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Hot Spot analysis output layer"
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output HTML report",
                fileFilter="HTML files (*.html)",
                optional=True
            )
        )
        self.addOutput(
            QgsProcessingOutputHtml(
                "HTML_REPORT_OUT",
                "Hot spot classification report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Retrieve parameters
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_getis_ord_report.html")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        correction = MULTIPLE_TESTING_OPTIONS[
            self.parameterAsEnum(parameters, self.MULTIPLE_TESTING, context)
        ][1]
        run_stability = self.parameterAsBoolean(parameters, self.STABILITY_ANALYSIS, context)

        # Validate target field
        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        field = source.fields().at(field_idx)
        if not field.isNumeric():
            raise QgsProcessingException(f"Target field '{field_name}' must be numeric.")

        check_weight_choice(source, weight_type, distance_band)
        feedback.pushInfo("Generating spatial weights matrix...")

        # Build weights matrix
        neighbors, weights, id_order, _ = build_weights_matrix(
            source,
            weight_type,
            k_neighbors=k_neighbors,
            distance_band=distance_band,
            feedback=feedback
        )
        stop_if_cancelled(feedback)

        feedback.pushInfo("Extracting target field values...")
        y_dict, skip_counts = read_finite_values(source, field_name, feedback)
        stop_if_cancelled(feedback)
        push_skips(feedback, skip_counts)

        valid_id_order = [fid for fid in id_order if fid in y_dict]
        no_data_ids = [fid for fid in id_order if fid not in y_dict]
        y = np.array([y_dict[fid] for fid in valid_id_order], dtype=float)

        if len(y) == 0:
            raise QgsProcessingException(
                "No finite numeric values found in the target field. " + skip_message(skip_counts)
            )

        def run_gi(raw_neighbors, raw_weights):
            """Gi* on the features with data; islands get NaN (no test)."""
            nbrs, wts, _ = restrict_weights(raw_neighbors, raw_weights, valid_id_order)
            counts = neighbour_counts(nbrs, valid_id_order)
            require_neighbours(nbrs, valid_id_order)
            with engine_errors():
                z, p, _ = calculate_getis_ord(y, nbrs, wts, valid_id_order, star=True)
            z = np.array(z, dtype=float)
            p = np.array(p, dtype=float)
            for i, fid in enumerate(valid_id_order):
                if counts[fid] == 0:
                    z[i] = p[i] = np.nan  # contract C1/C2: Gi* of a feature with itself only
            p_adj = adjust_p_values(p, correction)
            return z, p, p_adj, corrected_gi_bins(z, p_adj), nbrs, counts

        require_neighbours(restrict_weights(neighbors, weights, valid_id_order)[0], valid_id_order)
        feedback.pushInfo("Calculating Getis-Ord Gi* statistics...")
        z_scores, p_values, adjusted_p, conf_bins, neighbors, counts_by_fid = run_gi(neighbors, weights)
        isolated = np.array([counts_by_fid[fid] == 0 for fid in valid_id_order], dtype=bool)
        isolated_count = int(isolated.sum())
        stability_scores = np.where(isolated, np.nan, np.sign(conf_bins).astype(float))
        stability_runs = 1
        skipped_variants = []
        consensus_bins = conf_bins.copy()
        if run_stability:
            bin_runs = [conf_bins]
            variants = stability_variants(weight_type, k_neighbors, distance_band)
            feedback.pushInfo(
                f"Running Gi* stability analysis across {len(variants) + 1} neighborhood definitions..."
            )
            for variant_type, variant_k, variant_distance in variants:
                stop_if_cancelled(feedback)
                variant_neighbors, variant_weights, _, _ = build_weights_matrix(
                    source,
                    variant_type,
                    k_neighbors=variant_k,
                    distance_band=variant_distance,
                    feedback=feedback,
                )
                label = _variant_label(variant_type, variant_k, variant_distance)
                try:
                    bin_runs.append(run_gi(variant_neighbors, variant_weights)[3])
                except QgsProcessingException as exc:
                    # e.g. a 0.75x band under which every feature is an island:
                    # skip that definition, say so, and keep the others.
                    note = f"Stability run {label} was skipped: {exc}"
                    skipped_variants.append(note)
                    feedback.pushWarning(note)
            stop_if_cancelled(feedback)
            if len(bin_runs) == 1:
                raise QgsProcessingException(
                    "The stability consensus could not run: every alternative neighbourhood definition left "
                    "the features without neighbours. " + " ".join(skipped_variants)
                )
            stability_runs = len(bin_runs)
            directions = np.sign(np.vstack(bin_runs))
            stability_scores = np.mean(directions, axis=0)
            significant_runs = np.sum(directions != 0, axis=0)
            consensus_bins = np.where(
                (np.abs(stability_scores) >= 0.60) & (significant_runs >= 3),
                np.sign(stability_scores) * 2,
                0,
            ).astype(int)
            stability_scores = np.where(isolated, np.nan, stability_scores)
            consensus_bins = np.where(isolated, 0, consensus_bins)
            self.renderer_field = "gi_cons"
        self.consensus_map = bool(run_stability)
        missing = {NO_DATA: len(no_data_ids), NO_NEIGHBOURS: isolated_count}
        class_summary = getis_ord_class_summary(
            list(conf_bins) + [0] * len(no_data_ids), missing=missing,
        )
        feedback.pushInfo(class_summary["message"])
        stop_if_cancelled(feedback)

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("gi_zscore", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("gi_pvalue", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("gi_padj", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("gi_mt", QVariant.String, len=12))
        out_fields.append(QgsField("gi_conf", QVariant.Int))
        out_fields.append(QgsField("gi_class", QVariant.String, len=32))
        out_fields.append(QgsField("gi_nbrs", QVariant.Int))
        out_fields.append(QgsField("gi_stab", QVariant.Double, len=8, prec=4))
        out_fields.append(QgsField("gi_runs", QVariant.Int))
        out_fields.append(QgsField("gi_cons", QVariant.Int))

        # Initialize output sink
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        if sink is None:
            raise QgsProcessingException("Could not create the output layer.")

        def _num(value):
            return float(value) if np.isfinite(value) else None

        def _label(idx):
            if isolated[idx]:
                return NO_NEIGHBOURS
            if run_stability:
                c = int(consensus_bins[idx])
                return STABLE_HOT if c > 0 else STABLE_COLD if c < 0 else NOT_STABLE
            return GI_LABEL_BY_BIN[int(conf_bins[idx])]

        results_map = {}
        for idx, fid in enumerate(valid_id_order):
            results_map[fid] = (
                _num(z_scores[idx]), _num(p_values[idx]), _num(adjusted_p[idx]), int(conf_bins[idx]),
                _label(idx), counts_by_fid[fid], _num(stability_scores[idx]), int(consensus_bins[idx]),
            )
        if isolated_count:
            feedback.pushWarning(
                f"{isolated_count} feature(s) had no neighbour with data; they are written as "
                "'No neighbours' (gi_conf = 0, no z-score). Consider a larger distance band or K value."
            )

        feedback.pushInfo("Writing results to output layer...")
        mapped_counts = Counter()
        total = source.featureCount() or 1
        for current, f in enumerate(source.getFeatures()):
            stop_if_cancelled(feedback)

            out_feat = QgsFeature(f)
            set_fields_keeping_values(out_feat, out_fields)
            fid = f.id()
            out_feat.setAttribute("gi_mt", correction)
            out_feat.setAttribute("gi_runs", stability_runs)
            if fid in results_map:
                z, p, p_adjusted, c, label, neighbor_count, stability, consensus = results_map[fid]
                out_feat.setAttribute("gi_zscore", z)
                out_feat.setAttribute("gi_pvalue", p)
                out_feat.setAttribute("gi_padj", p_adjusted)
                out_feat.setAttribute("gi_conf", c)
                out_feat.setAttribute("gi_class", label)
                out_feat.setAttribute("gi_nbrs", int(neighbor_count))
                out_feat.setAttribute("gi_stab", stability)
                out_feat.setAttribute("gi_cons", consensus)
            else:
                label = NO_DATA
                out_feat.setAttribute("gi_zscore", None)
                out_feat.setAttribute("gi_pvalue", None)
                out_feat.setAttribute("gi_padj", None)
                out_feat.setAttribute("gi_conf", 0)
                out_feat.setAttribute("gi_class", NO_DATA)
                out_feat.setAttribute("gi_nbrs", None)
                out_feat.setAttribute("gi_stab", None)
                out_feat.setAttribute("gi_cons", 0)
            mapped_counts[label] += 1

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(50 + 50 * (current / total)))
        self.out_layer_id = dest_id

        feedback.pushInfo("Generating HTML report...")
        self._write_html(html_path, field_name, len(y), weight_type, class_summary, isolated_count, correction,
                         mapped_counts=mapped_counts, consensus=bool(run_stability),
                         stability_runs=stability_runs, skip_counts=skip_counts,
                         skipped_variants=skipped_variants)

        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _write_html(self, path, field_name, n, weight_type, class_summary, isolated_count, correction,
                    mapped_counts=None, consensus=False, stability_runs=1, skip_counts=None,
                    skipped_variants=None):
        # The donut shows exactly the classes drawn on the map (gi_class).
        mapped_counts = mapped_counts or {}
        style = (GI_CONSENSUS_STYLE if consensus else GI_CLASS_STYLE) + MISSING_CLASS_STYLE
        donut_labels = [label for _, _, label in style]
        donut_values = [int(mapped_counts.get(label, 0)) for _, _, label in style]
        donut_colors = {label: color for _val, color, label in style}
        method = correction_words(correction)
        no_data = int(class_summary.get("missing", {}).get(NO_DATA, 0))
        skipped_note = skip_message(skip_counts or {})
        consensus_note = ""
        if consensus:
            consensus_note = (
                f"The map shows the stability consensus over {int(stability_runs)} neighbourhood definitions: "
                f"{int(mapped_counts.get(STABLE_HOT, 0))} stable hot spot(s), "
                f"{int(mapped_counts.get(STABLE_COLD, 0))} stable cold spot(s), "
                f"{int(mapped_counts.get(NOT_STABLE, 0))} not stable. "
                "These are agreement classes, not confidence levels; gi_conf keeps the single-run confidence."
            )
            for note in skipped_variants or []:
                consensus_note += " " + note

        kpi_row = kpi_card_row_html([
            {"label": "Hot spot features", "value": str(class_summary["hot_count"]), "sublabel": "gi_conf > 0", "tone": "warn" if class_summary["hot_count"] else "neutral"},
            {"label": "Cold spot features", "value": str(class_summary["cold_count"]), "sublabel": "gi_conf < 0", "tone": "good" if class_summary["cold_count"] else "neutral"},
            {"label": "Significant total", "value": f"{class_summary['significant_count']} / {class_summary['tested_count']}", "sublabel": f"Dominant: {class_summary['dominant_label']}"},
            {"label": "Not tested", "value": str(no_data + int(isolated_count)), "sublabel": f"No data {no_data}, No neighbours {int(isolated_count)}", "tone": "warn" if (no_data or isolated_count) else "neutral"},
        ])

        guidance_html = analyst_guidance_html(
            "Getis-Ord Gi* Hot Spot Analysis",
            "Gi* tests whether each feature and its neighbors together form a statistically significant concentration of high (hot) or low (cold) values relative to the whole study area.",
            [
                "gi_nbrs is checked before trusting a 'Not Significant' result - zero valid neighbors looks identical to a genuine null result.",
                "The weight type (Queen/Rook/KNN/Distance Band) is a defensible neighborhood definition for this layer.",
                "A hot or cold spot forms a spatially coherent group rather than an isolated significant cell.",
            ],
            [
                f"{isolated_count} feature(s) had no neighbour with data and are mapped as 'No neighbours'" if isolated_count else "No isolated (zero-neighbor) features were found.",
                (skipped_note or "No feature was left out for a missing or non-finite value."),
                "A significant class appearing as a single isolated cell rather than a coherent cluster.",
                f"Multiple-testing correction: {method}; gi_conf uses gi_padj while gi_pvalue preserves the raw p-value.",
            ],
            [
                "Local Moran's I (LISA) for cluster AND outlier detection (HL/LH), not just hot/cold magnitude",
                "Incremental Spatial Autocorrelation to pick a defensible neighborhood scale first",
                "Spatial Gini / Colocation Quotient to quantify the concentration Gi* is flagging",
            ],
            "Use hot/cold spot classes to prioritize field verification or targeted intervention areas, not as a standalone causal claim about why a concentration exists.",
        )

        kpi_row_html = kpi_row
        donut_title = "Stability Consensus Breakdown" if consensus else "Hot/Cold Spot Class Breakdown"
        donut_html = donut_chart_svg(donut_labels, donut_values, colors=donut_colors, title=donut_title)
        caption = class_summary["message"] + (" " + consensus_note if consensus_note else "")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Hot Spot Analysis Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 760px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    section {{ margin: 28px 0; }}
    h2 {{ color: #1a202c; font-size: 1.15rem; margin: 0 0 12px 0; }}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
    {analyst_guidance_css()}
    {chart_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Hot Spot Analysis (Getis-Ord Gi*)</h1>
        <p class="subtitle">Field Analyzed: <strong>{html.escape(field_name)}</strong> | Features analysed: <strong>{int(n)}</strong> | Weights: <strong>{html.escape(weight_type)}</strong> | Multiple testing: <strong>{html.escape(method)}</strong></p>
    </header>

    {kpi_row_html}

    <section>
        <h2>{html.escape(donut_title)}</h2>
        {donut_html}
        <p class="chart-caption">{html.escape(caption)}</p>
    </section>

    {guidance_html}

    <footer>
        Generated by PlanX GeoStats Lab spatial statistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

    @style_loaded_output
    def postProcessAlgorithm(self, context, feedback):
        # Applies gorgeous hot/cold styling automatically in the QGIS GUI thread
        if self.out_layer_id is None:
            return {}

        layer = context.getMapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying Cold-to-Hot Hotspot symbology style...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats Getis-Ord Gi* hot spot output",
            {
                "gi_zscore": "Getis-Ord Gi* z-score",
                "gi_pvalue": "Getis-Ord Gi* p-value",
                "gi_padj": "Adjusted Gi* p-value used for confidence classification",
                "gi_mt": "Multiple-testing correction method",
                "gi_conf": "Hot/cold confidence class from -3 cold to +3 hot; 0 also for No data and No neighbours",
                "gi_class": "Class drawn on the map: confidence class, stability consensus class, No data or No neighbours",
                "gi_nbrs": "Valid neighbors used for the local statistic",
                "gi_stab": "Signed stability score across neighborhood definitions (-1 cold to +1 hot)",
                "gi_runs": "Number of neighborhood definitions in the stability analysis",
                "gi_cons": "Consensus hot/cold class; nonzero requires stable direction and at least three significant runs",
            },
            self.displayName(),
        )

        apply_renderer(layer, gi_class_renderer(layer.geometryType(), "gi_class",
                                                consensus=getattr(self, "consensus_map", False)))

        return {}
