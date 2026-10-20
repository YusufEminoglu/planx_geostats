# -*- coding: utf-8 -*-
"""S-maup test (Duque, Laniado & Polo 2018) Processing algorithm."""
from __future__ import annotations

import html

import numpy as np

from ..core.charts import kpi_card_row_html, line_chart_svg
from ..core.local_stats_engines import moran_global_perm, smaup_statistic
from ..core.reporting import analyst_guidance_html
from ._local_common import SpatialStatBase, ToolResult, callout, fmt, html_table, report_page


class SmaupAlgorithm(SpatialStatBase):
    FIELD = "FIELD"
    K_REGIONS = "K_REGIONS"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"

    ALG_ID = "smaup_test"
    DISPLAY = "S-maup: Is the MAUP Effect Significant?"
    GROUP = "02 | Urban Pattern Scan"
    GROUP_ID = "planx_pattern_scan"
    HAS_LAYER_OUTPUT = False
    REPORT_LABEL = "S-maup report"
    HELP = (
        "Before aggregating n areas into k larger zones - neighbourhoods into districts, blocks into "
        "service areas - the S-maup test of Duque, Laniado & Polo (2018) asks whether the statistics of the "
        "aggregated variable will depend on how the zones are drawn (the modifiable areal unit problem). "
        "The answer depends on three numbers only: n, k and the variable's spatial autocorrelation rho "
        "(Global Moran's I). Strong positive autocorrelation makes aggregation safe - neighbours are "
        "alike, so any grouping of them gives similar zone values - while weak or negative "
        "autocorrelation makes the result sensitive to the zoning.\n\n"
        "The tool measures rho on the input layer, evaluates the S-maup statistic for the k you plan to "
        "use, compares it with the critical values tabulated in the paper (the table row whose rho is "
        "nearest to the observed one, interpolated in n) and draws the curve over every k so you can see "
        "from which number of zones the effect becomes significant. The simulation study behind the table "
        "covers n = 25 to 900; outside that range the result is extrapolated and the report says so."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELD, "Variable to be aggregated")
        self.integer(self.K_REGIONS, "Number of zones you plan to aggregate into (k)", 10, minimum=1)

    def add_trailing_parameters(self):
        self.integer(self.PERMUTATIONS, "Permutations for Moran's I", 999, minimum=99, maximum=9999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)

    def compute(self, source, parameters, context, feedback):
        field = self.parameterAsString(parameters, self.FIELD, context)
        k = self.parameterAsInt(parameters, self.K_REGIONS, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        fids, x, dropped = self.read_matrix(source, [field], feedback)
        n = len(fids)
        self.require(n >= 4, "At least 4 features with a value are required.")
        self.require(k <= n, f"k = {k} exceeds the number of areas with a value ({n}).")
        self.require(x[:, 0].std() > 0, "The field is constant.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo("Measuring spatial autocorrelation (Global Moran's I)...")
        moran = moran_global_perm(x[:, 0], graph, permutations, seed)
        rho = float(np.clip(moran["I"], -0.99, 0.99))
        res = smaup_statistic(n, k, rho)
        ks = sorted(set(int(v) for v in np.unique(np.linspace(1, n, min(n, 400)).round())) | {k})
        curve = [smaup_statistic(n, kk, rho) for kk in ks]
        # S-maup falls as k grows: few large zones are the MAUP-sensitive case.
        # Report the largest k at which the effect is still significant.
        sig_ks = [kk for kk, r in zip(ks, curve) if r["level"] is not None and r["level"] <= 0.05]
        last_sig = max(sig_ks) if sig_ks else None
        chart = line_chart_svg(
            [float(v) for v in ks],
            {"S-maup": [r["smaup"] for r in curve],
             "5% critical value": [r["critical"][0.05] for r in curve]},
            x_label="number of zones k", y_label="S-maup", title="S-maup across the number of zones",
            markers=[(float(k), res["smaup"], f"k = {k}")],
        )
        significant = res["level"] is not None
        kpis = kpi_card_row_html([
            {"label": "S-maup", "value": fmt(res["smaup"], 4), "sublabel": f"n = {n}, k = {k}"},
            {"label": "Critical value (5%)", "value": fmt(res["critical"][0.05], 4)},
            {"label": "Moran's I (rho)", "value": fmt(moran["I"], 3), "sublabel": f"p = {fmt(moran['p'], 3)}"},
            {"label": "Verdict", "value": "MAUP-sensitive" if significant else "Robust",
             "tone": "bad" if significant else "good"},
        ])
        table = html_table(["Significance level", "Critical value", "S-maup exceeds it?"],
                           [[f"{int(a * 100)}%", res["critical"][a], "yes" if res["smaup"] > res["critical"][a] else "no"]
                            for a in (0.10, 0.05, 0.01)])
        notes = "".join(callout(html.escape(w)) for w in res["warnings"])
        if last_sig is not None and last_sig >= n:
            notes += callout("At this rho the MAUP effect is significant at the 5% level for every number of zones.",
                             info=True)
        elif last_sig is not None:
            notes += callout(f"At this rho the MAUP effect is significant at the 5% level for up to about "
                             f"<strong>k = {last_sig}</strong> zones and not beyond: aggregating into more zones than "
                             "that keeps the variable's distribution stable across alternative zonings (curve below).",
                             info=True)
        else:
            notes += callout("For no number of zones between 1 and n does S-maup exceed the 5% critical value at this rho.",
                             info=True)
        guidance = analyst_guidance_html(
            "S-maup",
            "S-maup estimates how much the distribution of the aggregated variable can change across alternative "
            "zonings with the same number of zones; above the critical value the change is significant.",
            [
                "Moran's I is measured with a neighbourhood that matches how zones will be built (contiguity for zoning).",
                "n lies inside 25-900, the range of the simulation study.",
                "The aggregation will group contiguous areas (S-maup assumes spatially contiguous zones).",
            ],
            [
                "A significant result does not say which zoning is right, only that the choice matters.",
                "The test uses a single autocorrelation number; a variable with different patterns in different parts "
                "of the study area is summarised crudely.",
                "rho is clipped to the table's -0.9..0.9 range of rows.",
            ],
            [
                "Global Moran's I for the autocorrelation itself",
                "Max-p Regions or SKATER to build zones that respect the variable",
                "Sensitivity Test to see how results move with the neighbourhood definition",
            ],
            "If the effect is significant, report results for more than one zoning, or build zones from the data "
            "(regionalisation) rather than adopting an arbitrary grid or administrative boundary.",
        )
        subtitle = (f"Variable: <strong>{html.escape(field)}</strong> | n = <strong>{n}</strong> | k = <strong>{k}</strong> | "
                    f"Weights: <strong>{html.escape(weight_label)}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        body = (kpis + drop + callout(html.escape(res["verdict"]) + f" (table row rho = {res['rho_row']:g})", info=True)
                + notes + f"<section><h2>Critical values</h2>{table}</section>"
                + f"<section><h2>How many zones are safe?</h2>{chart}</section>" + guidance)
        return ToolResult(report_html=report_page(self.DISPLAY, subtitle, body),
                          outputs={"SMAUP": res["smaup"], "CRITICAL_05": res["critical"][0.05], "MORANS_I": moran["I"]})
