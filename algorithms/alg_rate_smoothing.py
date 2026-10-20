# -*- coding: utf-8 -*-
"""Rate Smoothing (Empirical Bayes) Processing algorithm."""
from __future__ import annotations

import html

import numpy as np

from ..core.charts import bar_chart_svg, kpi_card_row_html, scatter_plot_svg
from ..core.local_stats_engines import rate_smoothing_summary
from ..core.reporting import analyst_guidance_html
from ._local_common import DOUBLE, SpatialStatBase, ToolResult, callout, fmt, html_table, report_page

RATE_METHODS = [
    ("Raw rate (events / base)", "rate_raw"),
    ("Global empirical Bayes (Marshall)", "rate_geb"),
    ("Spatial window rate (unit + neighbours)", "rate_sw"),
    ("Spatial empirical Bayes (local prior)", "rate_seb"),
    ("Assuncao-Reis EB-standardised rate (z-like)", "rate_arz"),
]


class RateSmoothingAlgorithm(SpatialStatBase):
    EVENTS = "EVENTS"
    BASE = "BASE"
    MAP_RATE = "MAP_RATE"
    MULTIPLIER = "MULTIPLIER"

    ALG_ID = "rate_smoothing"
    DISPLAY = "Rate Smoothing (Empirical Bayes)"
    GROUP = "01 | Data Preparation and Neighborhoods"
    GROUP_ID = "planx_prepare_neighbors"
    OUTPUT_LABEL = "Smoothed rates"
    REPORT_LABEL = "Rate smoothing report"
    HELP = (
        "Turns an event count and a population-at-risk field into rates that can be mapped and "
        "tested without small denominators driving the pattern. A unit with 40 residents and "
        "two incidents has a raw rate of 50 per 1000; a unit with 40 000 residents needs 2000 "
        "incidents for the same rate. The first number is mostly noise, but a raw-rate map shows "
        "both with the same colour and a hot-spot test treats both as equally reliable.\n\n"
        "Writes five rates side by side so the effect of each choice is visible:\n"
        "- rate_raw: events / base\n"
        "- rate_geb: global empirical Bayes (Marshall 1991) - each rate shrunk towards the "
        "study-area rate, more strongly the smaller its base\n"
        "- rate_sw: spatial window rate - events and bases of the unit plus its neighbours\n"
        "- rate_seb: spatial empirical Bayes - shrinkage towards the neighbourhood rate instead "
        "of the study-area rate\n"
        "- rate_arz: Assuncao-Reis (1999) EB-standardised rate - the variable to give Moran's I "
        "or a LISA when the question is about rates\n\n"
        "Rates are multiplied by the chosen multiplier (1000 = per thousand); rate_arz is a "
        "standardised score and is not multiplied. Features with a missing, zero or negative "
        "base are left out and written as NULL. Next step: Rate Cluster and Outlier Analysis "
        "(EB Local Moran) for clusters of high or low rates."
    )

    def add_parameters(self):
        self.numeric_field(self.EVENTS, "Event count field (numerator)")
        self.numeric_field(self.BASE, "Population at risk field (denominator)")
        self.enum(self.MAP_RATE, "Rate to style the map by", [label for label, _ in RATE_METHODS], 3)
        self.double(self.MULTIPLIER, "Rate multiplier (1000 = per thousand)", 1000.0, minimum=1e-9)

    def compute(self, source, parameters, context, feedback):
        events_field = self.parameterAsString(parameters, self.EVENTS, context)
        base_field = self.parameterAsString(parameters, self.BASE, context)
        multiplier = self.parameterAsDouble(parameters, self.MULTIPLIER, context)
        map_idx = self.parameterAsEnum(parameters, self.MAP_RATE, context)

        fids, matrix, dropped = self.read_matrix(source, [events_field, base_field], feedback)
        keep = (matrix[:, 1] > 0) & (matrix[:, 0] >= 0) if len(fids) else np.zeros(0, bool)
        bad_base = int(np.sum(~keep))
        fids = [fid for fid, k in zip(fids, keep) if k]
        matrix = matrix[keep]
        self.require(len(fids) >= 5, "At least 5 features with a valid event count and a positive base are required.")
        if bad_base:
            feedback.pushWarning(f"{bad_base} feature(s) had a zero/negative base or a negative count and were left out.")
        events, base = matrix[:, 0], matrix[:, 1]
        self.require(events.sum() > 0, "The event field sums to zero - there is no rate to smooth.")

        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo("Computing raw, empirical Bayes and spatial rates...")
        s = rate_smoothing_summary(events, base, graph)

        rows = {}
        for i, fid in enumerate(fids):
            rows[fid] = {
                "rate_raw": s["raw"][i] * multiplier,
                "rate_geb": s["global_eb"][i] * multiplier,
                "rate_sw": s["spatial"][i] * multiplier,
                "rate_seb": s["spatial_eb"][i] * multiplier,
                "rate_arz": s["assuncao_reis"][i],
                "eb_weight": s["global_weight"][i],
                "seb_weight": s["spatial_weight"][i],
            }
        fields = [(name, DOUBLE) for name in ("rate_raw", "rate_geb", "rate_sw", "rate_seb", "rate_arz", "eb_weight", "seb_weight")]
        docs = {
            "rate_raw": f"Raw rate x {multiplier:g}",
            "rate_geb": f"Global empirical Bayes rate x {multiplier:g} (Marshall 1991)",
            "rate_sw": f"Spatial window rate (unit + neighbours) x {multiplier:g}",
            "rate_seb": f"Spatial empirical Bayes rate x {multiplier:g}",
            "rate_arz": "Assuncao-Reis EB-standardised rate (unitless score)",
            "eb_weight": "Weight on the unit's own raw rate in the global EB estimate (0 = fully shrunk)",
            "seb_weight": "Weight on the unit's own raw rate in the spatial EB estimate",
        }
        report = self._report(events_field, base_field, multiplier, weight_label, s, base, dropped + bad_base, len(fids))
        return ToolResult(
            report_html=report, fields=fields, rows=rows, field_docs=docs,
            style=("quantile", RATE_METHODS[map_idx][1], 5, "#fff7ec", "#7f0000"),
            layer_title="PlanX GeoStats rate smoothing output",
        )

    def _report(self, events_field, base_field, multiplier, weight_label, s, base, excluded, n):
        raw = s["raw"] * multiplier
        geb = s["global_eb"] * multiplier
        seb = s["spatial_eb"] * multiplier
        kpis = kpi_card_row_html([
            {"label": "Units", "value": str(n), "sublabel": f"{excluded} excluded"},
            {"label": "Study-area rate", "value": fmt(s["overall_rate"] * multiplier), "sublabel": f"per {multiplier:g}"},
            {"label": "Raw-rate spread, smallest 20% bases", "value": fmt(s["raw_sd_small_base"] * multiplier),
             "sublabel": f"vs {fmt(s['raw_sd_large_base'] * multiplier)} for the largest 20%",
             "tone": "warn" if s["raw_sd_small_base"] > 2 * s["raw_sd_large_base"] else "neutral"},
            {"label": "Mean shrinkage weight", "value": fmt(float(np.mean(s["global_weight"])), 3),
             "sublabel": "1 = raw rate kept, 0 = study-area rate"},
        ])
        table = html_table(
            ["Rate", "Mean", "Std dev", "Min", "Max"],
            [[label, float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v))]
             for label, v in (("Raw", raw), ("Global EB", geb), ("Spatial window", s["spatial"] * multiplier),
                              ("Spatial EB", seb), ("Assuncao-Reis z", s["assuncao_reis"]))],
        )
        log_base = np.log10(base).tolist()
        scatter_raw = scatter_plot_svg(log_base, raw.tolist(), x_label="log10(base)", y_label=f"raw rate per {multiplier:g}",
                                       title="Raw rate against base size", trend_line=False)
        scatter_eb = scatter_plot_svg(log_base, geb.tolist(), x_label="log10(base)", y_label=f"global EB rate per {multiplier:g}",
                                      title="Global EB rate against base size", trend_line=False)
        rank_raw = np.argsort(np.argsort(-raw))
        rank_eb = np.argsort(np.argsort(-geb))
        top_raw = np.argsort(-raw)[:10]
        rows = [[f"#{i + 1}", base[i], raw[i], geb[i], int(rank_eb[i]) + 1] for i in top_raw]
        moved = int(np.sum(np.abs(rank_raw - rank_eb) > max(1, n // 10)))
        bars = bar_chart_svg([f"#{i + 1}" for i in top_raw], [float(raw[i] - geb[i]) for i in top_raw],
                             title="Shrinkage of the ten highest raw rates (raw minus EB)")
        clamp = ""
        if s["variance_clamped"]:
            clamp = callout(
                "The method-of-moments between-area variance came out negative "
                f"({fmt(s['between_variance'])}): the raw rates vary <em>less</em> than Poisson noise alone "
                "would make them. Every global EB rate has been set to the study-area rate - there is no "
                "detectable between-area signal at this scale. (esda's Empirical_Bayes would return negative "
                "shrinkage weights here; GeoDa and Marshall clamp to zero, as this tool does.)"
            )
        guidance = analyst_guidance_html(
            "Rate smoothing",
            "Empirical Bayes estimators borrow strength: a rate measured on a small base is pulled towards a prior "
            "(the study-area rate, or the neighbourhood rate) in proportion to how unreliable it is, while rates on "
            "large bases stay close to their raw value.",
            [
                "Events and base are counts of the same population (cases and the population that could become a case).",
                "The raw-rate spread is visibly larger on small bases than on large ones - the situation smoothing exists for.",
                "The spatial EB prior uses a neighbourhood large enough to hold a stable rate.",
            ],
            [
                f"{moved} unit(s) moved by more than a tenth of the ranking between raw and global EB rates - "
                "any league table built on raw rates is unstable.",
                "EB rates are shrunk towards a prior, so they understate genuine extremes on small bases; they are an "
                "estimate of the underlying risk, not a corrected count.",
                "Spatial window and spatial EB rates smooth across boundaries and can blur a sharp real edge.",
            ],
            [
                "Rate Cluster and Outlier Analysis (EB Local Moran) on the same two fields",
                "Global Moran's I on rate_arz rather than on rate_raw",
                "Hot Spot Analysis (Getis-Ord Gi*) on rate_seb for a smoothed hot-spot surface",
            ],
            "Report and rank smoothed rates, and keep the raw rate alongside for transparency; flag units whose ranking "
            "changes most as places where the raw number should not be quoted alone.",
        )
        subtitle = (f"Events: <strong>{html.escape(events_field)}</strong> | Base: <strong>{html.escape(base_field)}</strong> | "
                    f"Weights: <strong>{html.escape(weight_label)}</strong> | Multiplier: <strong>{multiplier:g}</strong>")
        body = (
            kpis
            + clamp
            + f"<section><h2>Rates side by side</h2>{table}</section>"
            + f"<section><h2>Why smoothing</h2>{scatter_raw}{scatter_eb}"
            + "<p class=\"chart-caption\">A funnel that narrows to the right in the first chart is small-number "
              "instability: the extreme raw rates sit on the smallest bases. Empirical Bayes pulls those points "
              "towards the study-area rate.</p></section>"
            + f"<section><h2>The ten highest raw rates</h2>{bars}"
            + html_table(["Unit (rank by raw rate)", "Base", "Raw rate", "Global EB rate", "Rank after EB"], rows)
            + "</section>"
            + guidance
        )
        return report_page(self.DISPLAY, subtitle, body)
