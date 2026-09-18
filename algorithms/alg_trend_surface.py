# -*- coding: utf-8 -*-
"""Polynomial Trend Surface Interpolation Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsRectangle,
)

from ..core.interpolation_engines import (
    NEIGHBOURHOODS,
    TREND_ORDERS,
    TREND_SOFT_LIMIT,
    TREND_TERM_COUNTS,
    cross_validate,
    cv_protocol_for,
    mean_nearest_distance,
    trend_fit,
    trend_surface,
    trend_terms,
)
from ..core.interpolation_report import (
    apply_hull_mask,
    cv_metric_rows,
    cv_protocol_note,
    fmt,
    hull_note,
    hull_summary,
    metric_rows,
    report_foot,
    report_head,
    surface_stats,
    validation_verdict,
)
from ..core.raster_io import (
    DEFAULT_NODATA,
    auto_cell_size,
    grid_centres,
    grid_shape,
    write_float32_geotiff,
)
from ..core.analysis_diagnostics import (
    caveats_html,
    diagnostics_html,
    push_diagnostics,
)
from ..core.reporting import analyst_guidance_html

from ._icons import algorithm_icon
from ._input import read_point_values, require_non_constant


logger = logging.getLogger("PlanX GeoStats Lab")

ORDER_LABELS = (
    "Order 1 - planar (3 coefficients)",
    "Order 2 - quadratic (6 coefficients)",
    "Order 3 - cubic (10 coefficients)",
)

NEIGHBOURHOOD_LABELS = (
    "Global - one fit over all observations",
    "Moving window - nearest N observations",
    "Moving window - observations within a radius",
)

#: The polynomial is refitted once per cell in the moving-window modes, so the
#: cost is linear in the cell count as well as in the observation count.  This
#: is the observation count above which leave-one-out validation stops being
#: affordable and k-fold takes over.
CV_LOO_LIMIT = 2000

#: The fewest observations each order can be fitted from.  One more than the
#: coefficient count, because a fit with as many observations as coefficients
#: has no residual degrees of freedom: it would pass through every point and
#: report a residual standard deviation of zero, which is not a fit.
MIN_OBSERVATIONS = {order: count + 1
                    for order, count in TREND_TERM_COUNTS.items()}


class TrendSurfaceAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    ORDER = "ORDER"
    NEIGHBOURHOOD = "NEIGHBOURHOOD"
    NEIGHBOURS = "NEIGHBOURS"
    RADIUS = "RADIUS"
    CELL_SIZE = "CELL_SIZE"
    MAX_POINTS = "MAX_POINTS"
    MASK_OUTSIDE = "MASK_OUTSIDE"
    RUN_CV = "RUN_CV"
    PREDICTION = "PREDICTION"
    STD_ERROR = "STD_ERROR"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "polynomial_trend_surface"

    def displayName(self) -> str:
        return "Polynomial Trend Surface Interpolation"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("polynomial_trend_surface")

    def createInstance(self):
        return TrendSurfaceAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits a polynomial surface of order 1, 2 or 3 to scattered point "
            "observations, either once over the whole layer or inside a moving "
            "neighbourhood around each cell.\n\n"
            "This is the interpolator to reach for when the field varies "
            "smoothly at a scale larger than the spacing of the observations - "
            "a regional gradient, a broad dome, a basin. It is the wrong tool "
            "for anything with local structure, because a polynomial cannot "
            "have local structure: raise the order to capture more detail and "
            "it starts to oscillate between the observations instead.\n\n"
            "The two modes answer different questions. A global fit gives one "
            "equation for the whole area - the classic trend surface - and its "
            "coefficients are readable statements about the field's overall "
            "shape, though they are in centred and scaled coordinates rather "
            "than the layer's own. A moving-window fit lets the polynomial "
            "describe a different local trend in each neighbourhood, which "
            "follows a changing gradient but stops being one surface you can "
            "summarise: the coefficients vary from cell to cell and only the "
            "mapped values are comparable.\n\n"
            "Order matters more than it looks. A cubic has ten coefficients and "
            "will happily produce a plausible surface from twenty observations "
            "that says more about those twenty points than about the field. "
            "Cross-validation is the test: if raising the order does not lower "
            "the error on held-out points, the extra terms are fitting the "
            "observations rather than the field, and the surface is worse for "
            "it even though it passes closer to the data.\n\n"
            "Outside the sampled area a polynomial does something a weighted "
            "average cannot: it keeps going. A cubic continued past the "
            "observations leaves the observed range within a few cells and "
            "keeps climbing or falling. The tool reports how many output cells "
            "fall outside the convex hull of the observations, and can write "
            "those cells as no-data, which is usually the honest choice.\n\n"
            "The optional standard error surface is an ordinary regression "
            "standard error. It describes how well the polynomial's "
            "coefficients are determined by the observations, and that is all: "
            "it does not include any spatial correlation, so it understates the "
            "real error between observations and grows without bound outside "
            "them. Read it as a comparison between places, never as a "
            "confidence interval.\n\n"
            "Output: a prediction raster (GeoTIFF Float32 covering the extent "
            "of the input points), an optional standard error raster, and an "
            "HTML report with the fitted coefficients, the cross-validation "
            "statistics and the extrapolation summary."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer (any geometry; centroids are used)",
                [QgsProcessing.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD,
                "Target numeric field to interpolate",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.ORDER,
                "Polynomial order",
                options=list(ORDER_LABELS),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.NEIGHBOURHOOD,
                "Fit",
                options=list(NEIGHBOURHOOD_LABELS),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NEIGHBOURS,
                "Neighbour count (moving-window nearest mode only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=24,
                minValue=1,
                maxValue=100_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS,
                "Radius in map units (moving-window radius mode only; 0 = five "
                "times the mean nearest-neighbour distance)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CELL_SIZE,
                "Output cell size in map units (0 = automatic, about 200 cells "
                "on the longer side)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_POINTS,
                "Maximum observations to fit from",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=TREND_SOFT_LIMIT,
                minValue=4,
                maxValue=2_000_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MASK_OUTSIDE,
                "Write cells outside the observations' convex hull as no-data "
                "(recommended for a polynomial)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_CV,
                "Run cross-validation (the only quality measure this method has)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.PREDICTION,
                "Predicted trend surface",
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.STD_ERROR,
                "Standard error of the fitted surface (not a confidence interval)",
                optional=True,
                createByDefault=False,
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
        self.addOutput(
            QgsProcessingOutputHtml(
                "HTML_REPORT_OUT", "Polynomial trend surface report"
            )
        )

    # ── run ──────────────────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        order = TREND_ORDERS[
            self.parameterAsEnum(parameters, self.ORDER, context)
        ]
        mode = NEIGHBOURHOODS[
            self.parameterAsEnum(parameters, self.NEIGHBOURHOOD, context)
        ]
        neighbours = self.parameterAsInt(parameters, self.NEIGHBOURS, context)
        radius = self.parameterAsDouble(parameters, self.RADIUS, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
        mask_outside = self.parameterAsBool(parameters, self.MASK_OUTSIDE, context)
        run_cv = self.parameterAsBool(parameters, self.RUN_CV, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_trend_surface_report.html"
            )

        feedback.pushInfo("Reading point locations and target values...")
        read = read_point_values(source, field_name, feedback)
        if read is None:
            return {}
        coords, values, numeric_summary, crs_warning = read
        push_diagnostics(feedback, numeric_summary, None, crs_warning)

        n = len(values)
        terms = TREND_TERM_COUNTS[order]
        floor = MIN_OBSERVATIONS[order]
        if n < floor:
            raise QgsProcessingException(
                f"A trend of order {order} has {terms} coefficients, so it "
                f"needs at least {floor} observations to leave any degrees of "
                f"freedom; got {n}. Use a lower order, or interpolate with a "
                "method that does not fit a global model."
            )
        require_non_constant(numeric_summary)
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. Raise the cap deliberately if "
                "you accept the wait, or reduce the point count first - a "
                "polynomial fit over a subset is a different surface, so the "
                "cross-validation score will tell you whether the subset "
                "changed the answer."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        spacing = mean_nearest_distance(coords_arr)
        k_arg = None
        radius_arg = None
        if mode == "nearest":
            k_arg = neighbours
            if k_arg <= terms:
                raise QgsProcessingException(
                    f"A moving-window fit of order {order} needs more than "
                    f"{terms} observations in each window; the neighbour count "
                    f"is {k_arg}. Cells with too few neighbours would be left "
                    "as no-data. Raise the neighbour count."
                )
            feedback.pushInfo(
                f"Refitting an order-{order} polynomial in the {k_arg} nearest "
                "observations around each cell."
            )
        elif mode == "radius":
            # Five spacings rather than three, because a polynomial needs more
            # observations per window than a weighted average does: with a
            # radius that reaches only the first ring, most windows would not
            # contain the term count and the grid would come back mostly empty.
            radius_arg = radius if radius > 0 else 5.0 * spacing
            feedback.pushInfo(
                f"Refitting an order-{order} polynomial within "
                f"{radius_arg:.6g} map units of each cell"
                + (f" (defaulted to five times the observation spacing of "
                   f"{spacing:.6g})" if radius <= 0 else "")
                + "."
            )
        else:
            feedback.pushInfo(
                f"Fitting a single order-{order} polynomial over all {n} "
                "observations."
            )

        if feedback.isCanceled():
            return {}

        # ── cross-validation, before the expensive part ──────────────────────
        cv = None
        cv_skip_note = None
        if run_cv:
            protocol, folds = cv_protocol_for(n, CV_LOO_LIMIT)
            feedback.pushInfo(
                f"Running {protocol} cross-validation over {n} observations..."
            )

            def _predict(train_coords, train_values, test_coords):
                out, _se, _used = trend_surface(
                    train_coords, train_values, test_coords, order, k_arg,
                    radius_arg,
                )
                return out

            try:
                cv = cross_validate(
                    _predict, coords_arr, values_arr,
                    protocol=protocol, folds=folds,
                )
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc
            except RuntimeError:
                return {}
            feedback.pushInfo(
                f"Cross-validation ({cv['protocol']}): ME={cv['me']:.6g}, "
                f"RMSE={cv['rmse']:.6g}, MAE={cv['mae']:.6g}"
            )
            if cv["n_dropped"]:
                cv_skip_note = (
                    f"{cv['n_dropped']} held-out observation(s) had too few "
                    "neighbours to fit the order without them and were excluded "
                    "from the statistics. Cells in the same position on the "
                    "output grid are written as no-data."
                )
                feedback.pushWarning(cv_skip_note)
            if feedback.isCanceled():
                return {}

        # ── output grid ──────────────────────────────────────────────────────
        extent = QgsRectangle(
            float(coords_arr[:, 0].min()), float(coords_arr[:, 1].min()),
            float(coords_arr[:, 0].max()), float(coords_arr[:, 1].max()),
        )
        if cell_size <= 0:
            cell_size = auto_cell_size(extent)
            feedback.pushInfo(
                f"Cell size defaulted to {cell_size:.6g} map units, giving about "
                "200 cells on the longer side."
            )
        try:
            cols, rows = grid_shape(extent, cell_size)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        res_x = extent.width() / cols
        res_y = extent.height() / rows
        cells = cols * rows
        feedback.pushInfo(
            f"Fitting a {cols} x {rows} grid ({cells:,} cells) at "
            f"{res_x:.6g} x {res_y:.6g} map units."
        )
        if mode != "all" and cells > 40_000:
            feedback.pushWarning(
                f"A moving-window fit refits the polynomial {cells:,} times, "
                "which can take several minutes at this grid size. Progress is "
                "reported as it goes and the run can be cancelled."
            )

        xs, ys = grid_centres(extent, cols, rows)
        gx, gy = np.meshgrid(xs, ys)
        targets = np.column_stack([gx.ravel(), gy.ravel()])

        def _progress(done, total):
            if feedback.isCanceled():
                return False
            feedback.setProgress(int(90.0 * done / max(1, total)))
            return True

        try:
            prediction, std_error, used = trend_surface(
                coords_arr, values_arr, targets, order, k_arg, radius_arg,
                progress=_progress,
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except RuntimeError:
            return {}

        if feedback.isCanceled():
            return {}

        prediction_grid = prediction.reshape(rows, cols)
        std_error_grid = std_error.reshape(rows, cols)
        used_grid = used.reshape(rows, cols).astype(float)

        unfitted = int((used == 0).sum())
        if unfitted:
            feedback.pushWarning(
                f"{unfitted:,} cell(s) had too few observations in their "
                "neighbourhood to fit an order-"
                f"{order} polynomial and were written as no-data. Widen the "
                "neighbourhood to fill them."
            )

        hull = hull_summary(coords_arr, targets)
        if hull is not None and hull["outside"]:
            feedback.pushInfo(
                f"{hull['outside']:,} of {hull['total']:,} cells "
                f"({hull['fraction']:.1%}) fall outside the convex hull of the "
                "observation locations."
            )

        write_grid = prediction_grid
        if mask_outside:
            write_grid = apply_hull_mask(prediction_grid, hull, DEFAULT_NODATA)
            feedback.pushInfo(
                "Cells outside the observation hull were masked to no-data."
            )

        feedback.pushInfo("Writing the trend surface...")
        pred_path = self.parameterAsOutputLayer(parameters, self.PREDICTION, context)
        if not pred_path:
            raise QgsProcessingException(
                "No destination was given for the trend surface."
            )
        try:
            write_float32_geotiff(pred_path, write_grid, extent,
                                  source.sourceCrs(), DEFAULT_NODATA,
                                  feedback=feedback)
        except (ValueError, RuntimeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

        se_path = self.parameterAsOutputLayer(parameters, self.STD_ERROR, context)
        # Mask the standard error with the same hull as the surface, so the two
        # rasters describe the same cells - and so the statistics reported below
        # describe the raster that was actually written rather than one with
        # more cells in it.
        se_write = std_error_grid
        if se_path:
            feedback.pushInfo("Writing the standard error surface...")
            if mask_outside:
                se_write = apply_hull_mask(std_error_grid, hull, DEFAULT_NODATA)
            try:
                write_float32_geotiff(se_path, se_write, extent,
                                      source.sourceCrs(), DEFAULT_NODATA,
                                      feedback=feedback)
            except (ValueError, RuntimeError) as exc:
                raise QgsProcessingException(str(exc)) from exc

        stats = {
            "prediction": surface_stats(write_grid),
            "std_error": surface_stats(se_write),
            "used": surface_stats(used_grid),
            "cols": cols,
            "rows": rows,
            "cells": cells,
            "res_x": float(res_x),
            "res_y": float(res_y),
            "unfitted": unfitted,
        }

        coefficients = None
        if mode == "all":
            coefficients = self.fit_global(coords_arr, values_arr, order,
                                           feedback)

        feedback.setProgress(95)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, order, mode, k_arg, radius_arg, spacing,
            cv, cv_skip_note, stats, numeric_summary, crs_warning, hull,
            mask_outside, se_path, max_points, coefficients,
        )

        result = {
            self.PREDICTION: pred_path,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        if se_path:
            result[self.STD_ERROR] = se_path
        feedback.setProgress(100)
        return result

    def fit_global(self, coords_arr, values_arr, order, feedback):
        """The single global fit, kept for the report's coefficient table."""
        try:
            state = trend_fit(coords_arr, values_arr, order)
        except ValueError as exc:
            feedback.pushWarning(
                f"The global fit could not be reported: {exc}"
            )
            return None
        return {
            "coefficients": [float(v) for v in state["coefficients"]],
            "terms": list(trend_terms(order)),
            "r2": state["r2"],
            "residual_std": state["residual_std"],
            "rank": state["rank"],
            "centre": [float(v) for v in state["centre"]],
            "scale": float(state["scale"]),
        }

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(
        self, path, field_name, n, order, mode, k_arg, radius_arg, spacing,
        cv, cv_skip_note, stats, numeric_summary, crs_warning, hull,
        mask_outside, se_path, max_points, coefficients,
    ):
        verdict, color, verdict_text, next_action = validation_verdict(
            cv, numeric_summary["std"] or 0.0, "The trend surface"
        )
        if cv is not None and cv["n_dropped"]:
            next_action += (
                f" {cv['n_dropped']} observation(s) could not be scored at all, "
                "which means the search neighbourhood is too small for this "
                "order."
            )
        if mode == "all" and order > 1:
            next_action += (
                f" Compare this order-{order} fit against order {order - 1} on "
                "the same points. If the lower order scores no worse on held-out "
                "observations, the extra terms are fitting the observations "
                "rather than the field."
            )

        pred_stats = stats["prediction"]
        se_stats = stats["std_error"]
        surface_pairs = [
            ("Output grid", f"{stats['cols']} x {stats['rows']} cells"),
            ("Cell size (map units)",
             f"{stats['res_x']:.6g} x {stats['res_y']:.6g}"),
            ("Predicted minimum", fmt(pred_stats["min"])),
            ("Predicted maximum", fmt(pred_stats["max"])),
            ("Predicted mean", fmt(pred_stats["mean"])),
            ("Observed minimum", fmt(numeric_summary["minimum"])),
            ("Observed maximum", fmt(numeric_summary["maximum"])),
            ("Observed mean", fmt(numeric_summary["mean"])),
            ("Cells written as no-data", f"{pred_stats['n_nodata']:,}"),
            ("Cells with too few neighbours to fit", f"{stats['unfitted']:,}"),
            ("Observation cap applied to this run", f"{max_points:,}"),
        ]

        if mode == "nearest":
            fit_txt = (
                f"moving window, {k_arg} nearest observations per cell"
            )
        elif mode == "radius":
            fit_txt = (
                f"moving window, observations within {radius_arg:.6g} map units"
            )
        else:
            fit_txt = "global, one fit over every observation"

        method_pairs = [
            ("Polynomial order", f"{order}"),
            ("Coefficients per fit", f"{TREND_TERM_COUNTS[order]}"),
            ("Fit", html.escape(fit_txt)),
            ("Mean nearest-neighbour distance (map units)", f"{spacing:.6g}"),
        ]

        coefficient_html = ""
        if coefficients is not None:
            coefficient_rows = metric_rows(
                (term, f"{value:.6g}")
                for term, value in zip(coefficients["terms"],
                                       coefficients["coefficients"])
            )
            r2_txt = fmt(coefficients["r2"])
            coefficient_html = f"""
    <section>
        <h2>Fitted Global Coefficients</h2>
        <table>
            <thead><tr><th>Term</th><th>Coefficient</th></tr></thead>
            <tbody>{coefficient_rows}</tbody>
        </table>
        <p>The coefficients are for coordinates centred on
        ({coefficients['centre'][0]:.6g}, {coefficients['centre'][1]:.6g}) and
        divided by {coefficients['scale']:.6g}, which is what keeps a cubic
        from being numerically singular on projected coordinates. They are
        therefore not on the layer's own scale and cannot be read as a
        coefficients-per-metre statement. The fit explains
        <strong>{r2_txt}</strong> of the variation in the observations and
        leaves a residual standard deviation of
        {coefficients['residual_std']:.6g}.</p>
    </section>
"""
        elif mode == "all":
            coefficient_html = """
    <section>
        <h2>Fitted Global Coefficients</h2>
        <p>The single global fit could not be reported; see the warnings on the
        run. The surface itself was still produced.</p>
    </section>
"""
        else:
            coefficient_html = """
    <section>
        <h2>Local Coefficients</h2>
        <p>This was a moving-window fit, so the polynomial was refitted once per
        cell and there is no single set of coefficients to report. Each window
        was fitted in the same centred and scaled coordinates as every other, so
        the mapped values are comparable across the grid, but the coefficients
        themselves vary from cell to cell and are not written out. If the
        coefficients are what you need, run the global fit.</p>
    </section>
"""

        se_note = (
            "The standard error surface was not written. It is the ordinary "
            "regression standard error of the fit: "
            f"its lowest value over the grid is {fmt(se_stats['min'])} and its "
            f"highest is {fmt(se_stats['max'])}."
            if not se_path else
            "The standard error raster is the ordinary regression standard "
            "error, sigma^2 * x'(X'X)^-1 x, evaluated at each cell. It measures "
            "how well the polynomial's coefficients are pinned down by where the "
            "observations happen to sit. It does not include any spatial "
            "correlation between observations, so it understates the real error "
            "between them and grows without bound outside them. Read it as a "
            "comparison between places within the mapped area, never as a "
            "confidence interval and never as an error bar on a cell."
        )

        cv_note = cv_protocol_note(cv)
        if cv_skip_note:
            cv_note += " " + cv_skip_note
        cv_note += (
            " A polynomial fit is not exact, so its residuals do measure "
            "something - but they measure how well the surface fits the data it "
            "was fitted from, which falls as the order rises and says nothing "
            "about held-out points. The numbers above come from refits that did "
            "not see the point being predicted."
        )

        extrapolation_html = (
            "<section>"
            "<h2>Outside the Observed Area</h2>"
            f"<p>{html.escape(hull_note(hull, mask_outside, DEFAULT_NODATA))}</p>"
            "</section>"
        )

        guidance_html = analyst_guidance_html(
            "Polynomial Trend Surface Interpolation",
            "Fits a polynomial of order 1 to 3 to the observations, globally or "
            "inside a moving window, to describe a field that varies smoothly.",
            [
                "The order was chosen by cross-validation, not by how closely "
                "the surface passes to the observations.",
                "The field varies at a scale larger than the spacing of the "
                "observations, which is what a polynomial can describe.",
                "Cells outside the observations' convex hull are masked or "
                "read as the method's behaviour away from data.",
            ],
            [
                "Raising the order to capture local detail. A polynomial has no "
                "local detail to give; the extra terms oscillate between "
                "observations instead.",
                "Reading a high-order fit as a trend. Past order 2 the "
                "coefficients stop having a physical reading and start "
                "describing the particular arrangement of the sample points.",
                "Using the standard error surface as a confidence interval. It "
                "excludes spatial correlation and is the smallest term of the "
                "real uncertainty.",
            ],
            [
                "Ordinary / Universal Kriging Interpolation, which fits a "
                "linear or quadratic drift and reports an analytic error "
                "surface that accounts for the observation geometry",
                "Local Moran's I to test whether the residuals from this trend "
                "cluster, which would mean the trend has left structure behind",
                "Geographically Weighted Regression if the goal becomes "
                "explaining the field rather than mapping its trend",
            ],
            "A polynomial trend surface answers where the field's broad shape "
            "goes, and by how much. Its residuals are the more useful product of "
            "the two: they are the field with the regional trend taken out, "
            "which is what the variogram tools want as input when a strong "
            "gradient would otherwise inflate the long-lag bins.",
        )

        caveat_text = [
            "A polynomial is a global mathematical form being asked to describe "
            "a field that was not generated by one. The fit is a summary of the "
            "observations, not a law they obey, and a good fit does not imply "
            "that the field continues that way anywhere else.",
            "A moving-window fit produces a surface whose local coefficients "
            "change from cell to cell. The mapped values are comparable across "
            "the grid because every window was fitted in the same coordinate "
            "system, but the surface is no longer one equation and cannot be "
            "summarised by one set of coefficients.",
            "Outside the convex hull of the observations a polynomial continues "
            "without constraint and will leave the observed range. Masking those "
            "cells is the honest default; leaving them as numbers produces a map "
            "that looks authoritative exactly where it has least support.",
            "The method carries no variance model. The standard error surface is "
            "a regression statistic about the coefficients, not a spatial "
            "prediction error, and the cross-validation score describes average "
            "behaviour over the observations rather than any single cell.",
        ]

        summary_txt = (
            f"An order-{order} polynomial ({TREND_TERM_COUNTS[order]} "
            f"coefficients) was fitted to {n} observations using a "
            f"{html.escape(fit_txt)}. The surface carries no variance model; its "
            "standard error raster is a regression statistic rather than a "
            "spatial prediction error. Cross-validation error was "
            f"<strong>{'not measured' if cv is None else fmt(cv['rmse'], 6)}</strong>"
            + (
                f", which is {cv['nrmse']:.3f} of the field's standard deviation "
                f"of {numeric_summary['std']:.6g}"
                if cv is not None and cv.get("nrmse") is not None
                else ""
            )
            + "."
        )

        head = report_head(
            "Polynomial Trend Surface Interpolation",
            f"Field Fitted: <strong>{html.escape(field_name)}</strong> | "
            f"Observations: <strong>{n}</strong> | Order: "
            f"<strong>{order}</strong> | Fit: "
            f"<strong>{html.escape(fit_txt)}</strong>",
            verdict, verdict_text, color,
        )

        body = f"""
    <section>
        <h2>Executive Summary</h2>
        <p>{summary_txt}</p>
    </section>

    <section>
        <h2>Method Settings</h2>
        <table>
            <thead><tr><th>Setting</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(method_pairs)}</tbody>
        </table>
    </section>

    {coefficient_html}

    <section>
        <h2>Cross-Validation</h2>
        <table>
            <thead><tr><th>Validation Statistic</th><th>Value</th></tr></thead>
            <tbody>{cv_metric_rows(cv)}</tbody>
        </table>
        <p>{html.escape(cv_note)}</p>
    </section>

    <section>
        <h2>Fitted Surface</h2>
        <table>
            <thead><tr><th>Surface Property</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(surface_pairs)}</tbody>
        </table>
        <div class="variance-box"><p style="margin:0;">{html.escape(se_note)}</p></div>
        <p style="margin-top:14px;">The surface was written as a GeoTIFF Float32
        raster covering the extent of the input points; cells that could not be
        fitted, or that fell outside the observation hull when masking was
        requested, carry the no-data value {DEFAULT_NODATA:g}.</p>
    </section>

    {extrapolation_html}

    {diagnostics_html(numeric_summary, None, crs_warning)}

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + body + report_foot(
                caveats_html("Polynomial trend surface", None,
                             numeric_summary, caveat_text),
                guidance_html,
            ))
