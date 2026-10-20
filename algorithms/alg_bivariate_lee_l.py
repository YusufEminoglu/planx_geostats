# -*- coding: utf-8 -*-
"""Bivariate Spatial Association (Lee's L, local) Processing algorithm.

Until v3.11.0 this tool computed z_x,i times the spatial lag of z_y - the
bivariate local Moran statistic, which Bivariate LISA already reports - under
the name Lee's L.  Lee's L smooths *both* variables, so it is symmetric in x
and y and carries the aspatial correlation as well as the spatial co-pattern.
It now computes the genuine statistic (validated against esda's
Spatial_Pearson_Local) and adds the permutation test it never had.
"""
from __future__ import annotations

import html
from collections import Counter

import numpy as np

from qgis.core import QgsProcessingParameterBoolean

from ..core.charts import donut_chart_svg, kpi_card_row_html, scatter_plot_svg
from ..core.local_pattern_audit import MULTIPLE_TESTING_OPTIONS, adjust_p_values
from ..core.local_stats_engines import lee_l
from ..core.reporting import analyst_guidance_html
from ..core.symbology import LISA_QUADRANT_STYLE
from ._local_common import permutation_resolution_warning, with_notes, DOUBLE, INT, STRING, SpatialStatBase, ToolResult, callout, fmt, report_page


class BivariateLeeLAlgorithm(SpatialStatBase):
    X_FIELD = "X_FIELD"
    Y_FIELD = "Y_FIELD"
    INCLUDE_SELF = "INCLUDE_SELF"
    PERMUTATIONS = "PERMUTATIONS"
    RANDOM_SEED = "RANDOM_SEED"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"

    ALG_ID = "bivariate_spatial_association_lees_l"
    DISPLAY = "Bivariate Spatial Association (Lee's L)"
    GROUP = "03 | Hot Spots and Spatial Outliers"
    GROUP_ID = "planx_hotspots_outliers"
    DEFAULT_WEIGHT = 2
    DEFAULT_KNN = 5
    OUTPUT_LABEL = "Output bivariate association layer"
    REPORT_LABEL = "Local Lee's L report"
    HELP = (
        "Local Lee's L (Lee 2001), also known as the local spatial Pearson statistic. Both fields are "
        "z-scored and spatially smoothed - each unit's value replaced by the average over its neighbourhood "
        "(including itself, as Lee recommends) - and the local statistic is the product of the two smoothed "
        "values: L_i = x~_i * y~_i. A large positive L_i means the neighbourhood is high (or low) on both "
        "fields together; a negative one means high on one and low on the other.\n\n"
        "Because both fields are smoothed, Lee's L is symmetric (swapping X and Y gives the same map) and "
        "combines two things: the plain correlation of X and Y, and how strongly each is spatially "
        "clustered. That separates it from Bivariate LISA, which smooths only Y and asks whether X at a "
        "location predicts Y around it.\n\n"
        "Significance: two-sided conditional permutation per unit, FDR correction by default. Output "
        "fields: lee_l, x_smooth, y_smooth, lee_p, lee_padj, lee_class (HH/LL/HL/LH by the signs of the "
        "smoothed values, kept only where significant), lee_nbrs.\n\n"
        "Changed in 3.11.0: earlier versions computed z_x times the lag of z_y (the bivariate Moran "
        "statistic) under this name, with no significance test; y_lag_z is replaced by x_smooth/y_smooth."
    )

    def add_parameters(self):
        self.numeric_field(self.X_FIELD, "Primary numeric field (X)")
        self.numeric_field(self.Y_FIELD, "Second numeric field (Y)")

    def add_trailing_parameters(self):
        self.addParameter(QgsProcessingParameterBoolean(
            self.INCLUDE_SELF, "Include each unit in its own smoothing window (Lee 2001)", defaultValue=True))
        self.integer(self.PERMUTATIONS, "Conditional permutations", 999, minimum=99, maximum=9999)
        self.integer(self.RANDOM_SEED, "Random seed", 42, minimum=0)
        self.enum(self.MULTIPLE_TESTING, "Multiple-testing correction", [label for label, _ in MULTIPLE_TESTING_OPTIONS], 1)

    def compute(self, source, parameters, context, feedback):
        x_field = self.parameterAsString(parameters, self.X_FIELD, context)
        y_field = self.parameterAsString(parameters, self.Y_FIELD, context)
        self.require(x_field != y_field, "Choose two different fields for bivariate association.")
        include_self = self.parameterAsBool(parameters, self.INCLUDE_SELF, context)
        permutations = self.parameterAsInt(parameters, self.PERMUTATIONS, context)
        seed = self.parameterAsInt(parameters, self.RANDOM_SEED, context)
        correction = MULTIPLE_TESTING_OPTIONS[self.parameterAsEnum(parameters, self.MULTIPLE_TESTING, context)][1]
        fids, x, dropped = self.read_matrix(source, [x_field, y_field], feedback)
        self.require(len(fids) >= 4, "At least 4 complete numeric records are required.")
        self.require(np.std(x[:, 0]) > 0 and np.std(x[:, 1]) > 0, "Both selected fields must vary across complete records.")
        graph, weight_label = self.read_weights(source, parameters, context, feedback, fids)
        feedback.pushInfo("Computing Lee's L (global and local)...")
        res = lee_l(x[:, 0], x[:, 1], graph, permutations, seed, include_self)
        p_adj = adjust_p_values(np.nan_to_num(res["local_p"], nan=1.0), correction)
        p_adj = np.where(np.isfinite(res["local_p"]), p_adj, np.nan)
        classes = [c if np.isfinite(q) and q < 0.05 else "Not Significant" for c, q in zip(res["local_class"], p_adj)]
        rows = {fid: {"lee_l": res["local"][i], "x_smooth": res["smoothed_x"][i], "y_smooth": res["smoothed_y"][i],
                      "lee_p": res["local_p"][i], "lee_padj": p_adj[i], "lee_class": classes[i],
                      "lee_nbrs": int(graph.cardinality[i])} for i, fid in enumerate(fids)}
        fields = [("lee_l", DOUBLE), ("x_smooth", DOUBLE), ("y_smooth", DOUBLE), ("lee_p", DOUBLE),
                  ("lee_padj", DOUBLE), ("lee_class", STRING), ("lee_nbrs", INT)]
        docs = {
            "lee_l": "Local Lee's L: product of the smoothed z-scores of X and Y",
            "x_smooth": f"Spatially smoothed z-score of {x_field}",
            "y_smooth": f"Spatially smoothed z-score of {y_field}",
            "lee_p": "Two-sided conditional-permutation p-value",
            "lee_padj": f"p-value after {correction} correction",
            "lee_class": "HH/LL/HL/LH by the signs of the smoothed values (significant units only)",
            "lee_nbrs": "Neighbours used",
        }
        report = self._report(x_field, y_field, include_self, weight_label, permutations, correction, res, classes, dropped)
        warn = permutation_resolution_warning(permutations, int(np.isfinite(res["local_p"]).sum()), correction)
        if warn:
            feedback.pushWarning(warn)
        report = with_notes(report, [warn])
        return ToolResult(report_html=report, fields=fields, rows=rows, field_docs=docs,
                          style=("lisa", "lee_class"), layer_title="PlanX GeoStats Lee's L output")

    def _report(self, x_field, y_field, include_self, weight_label, permutations, correction, res, classes, dropped):
        counts = Counter(classes)
        kpis = kpi_card_row_html([
            {"label": "Global Lee's L", "value": fmt(res["L"], 3), "sublabel": f"p = {fmt(res['p'], 3)}",
             "tone": "good" if res["p"] < 0.05 else "neutral"},
            {"label": "Pearson r (raw)", "value": fmt(res["pearson_r"], 3)},
            {"label": "r of smoothed fields", "value": fmt(res["smoothed_r"], 3)},
            {"label": "Significant units", "value": str(len(classes) - counts["Not Significant"])},
        ])
        colors = {label: color for _code, color, label in LISA_QUADRANT_STYLE}
        donut = donut_chart_svg([label for _c, _col, label in LISA_QUADRANT_STYLE],
                                [counts.get(code, 0) for code, _col, _l in LISA_QUADRANT_STYLE],
                                colors=colors, title="Local Lee's L classes")
        scatter = scatter_plot_svg(res["smoothed_x"].tolist(), res["smoothed_y"].tolist(),
                                   x_label=f"smoothed z({x_field})", y_label=f"smoothed z({y_field})",
                                   title="Smoothed X against smoothed Y", quadrant_shading=True)
        guidance = analyst_guidance_html(
            "Local Lee's L",
            "Where both smoothed values are high (HH) or both low (LL), the two fields co-cluster: the neighbourhood is "
            "consistently high or low on both.",
            ["Both fields are spatially structured on their own (check Global Moran's I).",
             "Significant units form groups.", "The smoothing window matches the scale of the question."],
            ["Lee's L includes the aspatial correlation: two strongly correlated fields give a large L even with weak "
             "spatial clustering - compare the r of the raw and the smoothed fields.",
             "Smoothing blurs sharp boundaries; a narrow feature can vanish."],
            ["Global Bivariate Lee's L for the study-area summary",
             "Bivariate LISA for the asymmetric question (does X here predict Y around it?)",
             "Local Moran's I on each field separately"],
            "Use HH/LL co-clusters to find areas where two conditions coincide at the neighbourhood scale, for example "
            "high density together with high accessibility.",
        )
        subtitle = (f"X: <strong>{html.escape(x_field)}</strong> | Y: <strong>{html.escape(y_field)}</strong> | Weights: "
                    f"<strong>{html.escape(weight_label)}</strong> (binary{', self included' if include_self else ''}, row-standardised) | "
                    f"Permutations: <strong>{permutations}</strong> | Multiple testing: <strong>{html.escape(correction)}</strong>")
        drop = callout(f"{dropped} feature(s) with missing values were left out.") if dropped else ""
        body = (kpis + drop + f"<section><h2>Classes</h2>{donut}</section><section><h2>Smoothed values</h2>{scatter}</section>"
                + guidance)
        return report_page(self.DISPLAY, subtitle, body)
