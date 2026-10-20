# -*- coding: utf-8 -*-
"""Global Bivariate Lee's L (spatial Pearson) Processing algorithm.

Until v3.11.0 this computed the mean of z_x,i times the spatial lag of z_y -
the global bivariate Moran statistic - under Lee's name.  It now computes
Lee's (2001) L, which smooths both variables, together with the spatial
smoothing scalars L_XX and L_YY; the value matches esda's Spatial_Pearson.
"""
from __future__ import annotations

import html

import numpy as np

from qgis.core import QgsProcessingParameterBoolean

from ..core.charts import kpi_card_row_html, scatter_plot_svg
from ..core.local_stats_engines import lee_l
from ..core.reporting import analyst_guidance_html
from ._local_common import SpatialStatBase, ToolResult, callout, fmt, html_table, report_page


class GlobalLeesLAlgorithm(SpatialStatBase):
    FIELD_X = "FIELD_X"
    FIELD_Y = "FIELD_Y"
    INCLUDE_SELF = "INCLUDE_SELF"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"

    ALG_ID = "global_bivariate_lee_l"
    DISPLAY = "Global Bivariate Lee's L"
    GROUP = "02 | Urban Pattern Scan"
    GROUP_ID = "planx_pattern_scan"
    DEFAULT_KNN = 5
    HAS_LAYER_OUTPUT = False
    REPORT_LABEL = "Global Bivariate Lee's L diagnostic report"
    HELP = (
        "Lee's (2001) L for two fields over the whole study area - also called the spatial Pearson "
        "coefficient. Each field is z-scored and spatially smoothed with the neighbourhood average "
        "(including the unit itself, as Lee recommends); L is the average product of the two smoothed "
        "fields. It combines two things a Pearson correlation alone cannot: whether the fields go "
        "together, and whether each is spatially clustered. Lee shows L = sqrt(L_XX) * sqrt(L_YY) * "
        "r(x~, y~), where L_XX and L_YY - the spatial smoothing scalars, also reported - measure how "
        "clustered each field is on its own.\n\n"
        "Two permutation tests are reported. Permuting the (X, Y) pairs over locations keeps their "
        "plain correlation and asks whether the spatial co-pattern goes beyond it; shuffling Y alone "
        "against a fixed X asks whether there is any bivariate spatial association at all.\n\n"
        "Run Bivariate Spatial Association (Lee's L) on the same pair to see where the co-clustering "
        "sits. Changed in 3.11.0: earlier versions computed the global bivariate Moran statistic under "
        "this name; values are not comparable with results from 3.10 and before."
    )

    def add_parameters(self):
        self.numeric_field(self.FIELD_X, "Field X")
        self.numeric_field(self.FIELD_Y, "Field Y")

    def add_trailing_parameters(self):
        self.addParameter(QgsProcessingParameterBoolean(
            self.INCLUDE_SELF, "Include each unit in its own smoothing window (Lee 2001)", defaultValue=True))
        self.integer(self.PERMUTATIONS, "Number of permutations", 999, minimum=99, maximum=99999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)

    def compute(self, source, parameters, context, feedback):
        fx = self.parameterAsString(parameters, self.FIELD_X, context)
        fy = self.parameterAsString(parameters, self.FIELD_Y, context)
        self.require(fx != fy, "Field X and Field Y must be different fields.")
        include_self = self.parameterAsBool(parameters, self.INCLUDE_SELF, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        fids, x, dropped = self.read_matrix(source, [fx, fy], feedback)
        self.require(len(fids) >= 4, "At least 4 valid features with both fields populated are required.")
        self.require(np.std(x[:, 0]) > 0 and np.std(x[:, 1]) > 0,
                     "Both fields must have variation; a constant field cannot be spatially co-clustered.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo(f"Calculating Lee's L with {permutations} permutations...")
        res = lee_l(x[:, 0], x[:, 1], graph, permutations, seed, include_self)
        feedback.pushInfo(f"Lee's L = {res['L']:.4f}, z = {res['z']:.3f}, p = {res['p']:.4f}")

        significant = res["p"] < 0.05
        if significant and res["L"] > res["sim_mean"]:
            verdict = ("Positive spatial co-clustering: neighbourhoods that are high on one field are high on the other, "
                       "beyond what the plain correlation of the two fields implies.")
        elif significant:
            verdict = "Negative spatial co-clustering: neighbourhoods high on one field are low on the other."
        else:
            verdict = "No spatial co-pattern beyond the plain correlation of the two fields."
        kpis = kpi_card_row_html([
            {"label": "Lee's L", "value": fmt(res["L"], 4), "sublabel": f"z = {fmt(res['z'], 3)}"},
            {"label": "p (pairs permuted)", "value": fmt(res["p"], 3), "tone": "good" if significant else "neutral"},
            {"label": "p (Y shuffled)", "value": fmt(res["p_any"], 3)},
            {"label": "Pearson r", "value": fmt(res["pearson_r"], 3)},
        ])
        table = html_table(
            ["Quantity", "Value", "Meaning"],
            [
                ["Lee's L", res["L"], "Average product of the smoothed z-scores"],
                ["L_XX", res["L_xx"], f"Spatial smoothing scalar of {fx} (1 = perfectly clustered, ~1/k = random)"],
                ["L_YY", res["L_yy"], f"Spatial smoothing scalar of {fy}"],
                ["r(x, y)", res["pearson_r"], "Plain Pearson correlation"],
                ["r(x~, y~)", res["smoothed_r"], "Correlation of the smoothed fields"],
                ["Permutation mean (pairs)", res["sim_mean"], "Expected L given the correlation, without spatial pattern"],
                ["Permutation sd (pairs)", res["sim_sd"], ""],
            ],
        )
        scatter = scatter_plot_svg(res["smoothed_x"].tolist(), res["smoothed_y"].tolist(),
                                   x_label=f"smoothed z({fx})", y_label=f"smoothed z({fy})",
                                   title="Smoothed X against smoothed Y", quadrant_shading=True)
        guidance = analyst_guidance_html(
            "Global Bivariate Lee's L",
            "Lee's L is large when the two fields are correlated and each is spatially clustered, so that whole "
            "neighbourhoods are high (or low) on both.",
            ["Both fields are measured on the same units and are meaningful to average over neighbours.",
             "Each field shows its own spatial structure (Global Moran's I).",
             "The two p-values tell the same story, or the difference is understood."],
            ["Do not read L as a correlation coefficient: it is bounded by sqrt(L_XX * L_YY), not by 1.",
             "A significant 'Y shuffled' p with a non-significant 'pairs' p means the association is the plain "
             "correlation - there is no extra spatial co-pattern.",
             "Results from versions before 3.11 measured a different statistic."],
            ["Bivariate Spatial Association (Lee's L) to map where the co-clustering is",
             "Bivariate LISA for the asymmetric question", "Global Moran's I on each field"],
            "Use Lee's L to decide whether two urban conditions coincide at the neighbourhood scale before mapping "
            "the local co-clusters.",
        )
        subtitle = (f"Field X: <strong>{html.escape(fx)}</strong> | Field Y: <strong>{html.escape(fy)}</strong> | N: "
                    f"<strong>{len(fids)}</strong> | Weights: <strong>{html.escape(weight_label)}</strong> (binary"
                    f"{', self included' if include_self else ''}) | Permutations: <strong>{permutations}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        body = (kpis + drop + callout(html.escape(verdict), info=True)
                + f"<section><h2>Decomposition</h2>{table}</section><section><h2>Smoothed fields</h2>{scatter}</section>"
                + guidance)
        return ToolResult(report_html=report_page(self.DISPLAY, subtitle, body),
                          outputs={"LEES_L": res["L"], "P_VALUE": res["p"]})
