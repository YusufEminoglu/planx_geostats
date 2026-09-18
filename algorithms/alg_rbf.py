# -*- coding: utf-8 -*-
"""Radial Basis Function Interpolation Processing Algorithm."""
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
    RBF_KERNELS,
    RBF_LOO_LIMIT,
    RBF_SOFT_LIMIT,
    cross_validate,
    cv_protocol_for,
    iter_chunks,
    mean_nearest_distance,
    rbf_predict,
    rbf_weights,
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

KERNEL_LABELS = (
    "Thin plate spline (r^2 ln r)",
    "Multiquadric",
    "Inverse multiquadric",
    "Gaussian",
)

#: Kernels whose shape parameter is a distance the user can set.  Thin plate
#: has no shape parameter at all, which is why it is not in this list: its form
#: is fixed, and the "shape" control is ignored for it rather than silently
#: accepted.
SHAPED_KERNELS = ("multiquadric", "inverse_multiquadric", "gaussian")

#: Leave-one-out is a genuine refit for a radial basis function - the system
#: changes when a point is removed, not merely its right-hand side - so each
#: held-out point costs a full solve.  Above this the validation switches to
#: k-fold, which fits on less data and therefore scores worse; the report says
#: so, because the two numbers are not comparable.
CV_LOO_LIMIT = RBF_LOO_LIMIT

KERNEL_NOTES = {
    "thin_plate": (
        "The thin plate spline is the kernel that minimises bending energy, "
        "which is why it is the classic choice for a smooth surface through "
        "scattered points. It is the only kernel here with a polynomial term "
        "built into its system, and that term is what gives it linear "
        "precision: if the observations lie on a plane, the surface reproduces "
        "that plane exactly. It has no shape parameter - its form is fixed, so "
        "the shape control is ignored for it."
    ),
    "multiquadric": (
        "The multiquadric kernel produces a surface that is smooth and, at "
        "large distances, grows rather than flattens. It is well behaved "
        "between the observations but its extrapolation is unbounded in a way "
        "the thin plate spline's is not. It has no polynomial term, so it does "
        "not reproduce a plane exactly - a genuine difference from the thin "
        "plate spline, not a numerical one."
    ),
    "inverse_multiquadric": (
        "The inverse multiquadric has the same shape as the multiquadric "
        "inverted: it falls off with distance instead of growing, so the "
        "surface flattens away from the observations rather than running away. "
        "That makes it the more forgiving of the two outside the sampled area, "
        "at the cost of a flatter surface within it."
    ),
    "gaussian": (
        "The gaussian kernel falls off fastest of the four, so each "
        "observation's influence is local. It produces the smoothest surface "
        "here and the most sensitive one to the shape parameter: set the shape "
        "too small and the surface becomes a set of isolated bumps, too large "
        "and it becomes featureless."
    ),
}


class RbfAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    KERNEL = "KERNEL"
    SHAPE = "SHAPE"
    SMOOTHING = "SMOOTHING"
    CELL_SIZE = "CELL_SIZE"
    MAX_POINTS = "MAX_POINTS"
    MASK_OUTSIDE = "MASK_OUTSIDE"
    RUN_CV = "RUN_CV"
    PREDICTION = "PREDICTION"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "rbf_interpolation"

    def displayName(self) -> str:
        return "Radial Basis Function Interpolation (RBF)"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("rbf_interpolation")

    def createInstance(self):
        return RbfAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Predicts a smooth continuous surface from scattered point "
            "observations by fitting a sum of radial basis functions, one "
            "centred on each observation, that passes through every point "
            "exactly.\n\n"
            "Where inverse distance weighting averages its neighbours, a radial "
            "basis function solves a linear system for the coefficients of a "
            "kernel centred on each observation. The result is the smoothest "
            "surface of its kind that honours the data - smooth in the specific "
            "sense of minimising bending energy, for the thin plate spline - "
            "rather than smooth because a neighbourhood was made wide.\n\n"
            "The kernel is the whole choice. Thin plate spline is the classic "
            "and the only one here with a linear polynomial term built into its "
            "system, which gives it linear precision: observations lying on a "
            "plane are reproduced exactly. Multiquadric, inverse multiquadric "
            "and gaussian have no such term and therefore no such property - "
            "that is a real mathematical difference between the kernels, not a "
            "numerical one, and the report states it for the kernel you picked. "
            "The three shaped kernels take a shape parameter, defaulted from "
            "the mean distance between neighbouring observations; thin plate "
            "takes none.\n\n"
            "Smoothing adds a ridge to the system, giving up exactness at the "
            "observations in exchange for tolerance of noise in them. At zero - "
            "the default - the surface passes through every observation and the "
            "reported residual is zero to rounding. Any non-zero smoothing "
            "makes the residual non-zero, and that residual is the honest "
            "statement of how much exactness was traded away. Smoothing a "
            "surface because it looks too wiggly is reasonable; doing it and "
            "then reporting the surface as an exact interpolation is not, which "
            "is why the residual is reported either way.\n\n"
            "The cost is a dense solve whose size grows with the square of the "
            "observation count, so this tool caps observations far lower than "
            "the other interpolators. Cross-validation is the only quality "
            "measure available - the kernel carries no variance model. Because "
            "removing an observation changes the system and not just its "
            "right-hand side, leave-one-out validation genuinely refits once "
            f"per observation, which is why above {RBF_LOO_LIMIT} observations "
            "it is replaced by k-fold. Those two numbers are not comparable and "
            "the report says which was used.\n\n"
            "Output: a prediction raster (GeoTIFF Float32 covering the extent "
            "of the input points) and an HTML report with the cross-validation "
            "statistics, the system's conditioning and residual, and how much "
            "of the output the observations do not span."
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
                self.KERNEL,
                "Radial basis kernel",
                options=list(KERNEL_LABELS),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SHAPE,
                "Kernel shape parameter in map units (0 = derive from the mean "
                "nearest-neighbour distance; ignored by thin plate)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SMOOTHING,
                "Smoothing (0 = interpolate exactly through every observation)",
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
                "Maximum observations (the solve is dense; its cost grows with "
                "the square of this number)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=RBF_SOFT_LIMIT,
                minValue=3,
                maxValue=200_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MASK_OUTSIDE,
                "Write cells outside the observations' convex hull as no-data",
                defaultValue=False,
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
                "Predicted surface",
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
            QgsProcessingOutputHtml("HTML_REPORT_OUT", "RBF interpolation report")
        )

    # ── run ──────────────────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        kernel = RBF_KERNELS[
            self.parameterAsEnum(parameters, self.KERNEL, context)
        ]
        shape = self.parameterAsDouble(parameters, self.SHAPE, context)
        smoothing = self.parameterAsDouble(parameters, self.SMOOTHING, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
        mask_outside = self.parameterAsBool(parameters, self.MASK_OUTSIDE, context)
        run_cv = self.parameterAsBool(parameters, self.RUN_CV, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_rbf_report.html"
            )

        feedback.pushInfo("Reading point locations and target values...")
        read = read_point_values(source, field_name, feedback)
        if read is None:
            return {}
        coords, values, numeric_summary, crs_warning = read
        push_diagnostics(feedback, numeric_summary, None, crs_warning)

        n = len(values)
        if n < 3:
            raise QgsProcessingException(
                f"Radial basis interpolation needs at least 3 observations with "
                f"a finite value and a usable geometry; got {n}."
            )
        require_non_constant(numeric_summary)
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. The radial basis system is "
                "dense, so its cost grows with the square of the observation "
                "count and the solve becomes the run. Raise the cap deliberately "
                "if you accept the wait, or reduce the point count first."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        spacing = mean_nearest_distance(coords_arr)
        shape_arg = None
        if kernel == "thin_plate":
            if shape > 0:
                feedback.pushWarning(
                    "The shape parameter is ignored by the thin plate spline, "
                    "whose form is fixed. Use one of the shaped kernels if you "
                    "need to control the surface's scale of variation."
                )
        else:
            shape_arg = shape if shape > 0 else spacing
            if shape > 0:
                feedback.pushInfo(
                    f"Kernel shape parameter: {shape_arg:.6g} map units."
                )
            else:
                feedback.pushInfo(
                    f"Kernel shape parameter defaulted to {shape_arg:.6g} map "
                    f"units, the mean nearest-neighbour distance."
                )
            if shape > 0 and not (0.1 * spacing <= shape <= 10.0 * spacing):
                feedback.pushWarning(
                    f"The shape parameter ({shape:.6g}) is far from the mean "
                    f"observation spacing ({spacing:.6g}). The gaussian and "
                    "inverse multiquadric kernels are the sensitive ones: too "
                    "small gives isolated bumps around each observation, too "
                    "large gives a surface with no local detail. Compare "
                    "cross-validation scores rather than judging by eye."
                )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"Solving the {kernel} system for {n} observations"
            + (f" with smoothing {smoothing:g}" if smoothing > 0 else "")
            + "..."
        )
        try:
            state = rbf_weights(coords_arr, values_arr, kernel, shape_arg,
                                smoothing)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        if smoothing > 0 and state["max_abs_residual"] > 0:
            feedback.pushInfo(
                f"Smoothing {smoothing:g} gives up exactness: RMS residual "
                f"{state['rms_residual']:.6g}, largest single residual "
                f"{state['max_abs_residual']:.6g}."
            )
        else:
            feedback.pushInfo(
                "The surface passes through every observation exactly "
                "(residual zero to rounding)."
            )

        # ── cross-validation, before the expensive part ──────────────────────
        cv = None
        cv_skip_note = None
        if run_cv:
            protocol, folds = cv_protocol_for(n, CV_LOO_LIMIT)
            if protocol == "kfold":
                cv_skip_note = (
                    f"Leave-one-out validation was not affordable at {n} "
                    f"observations, above the limit of {CV_LOO_LIMIT}. Each "
                    "held-out point needs its own solve of the whole system, so "
                    f"k-fold validation was used instead. Its error is "
                    "systematically larger than a leave-one-out figure would "
                    "have been, and the two must not be compared."
                )
                feedback.pushWarning(cv_skip_note)
            feedback.pushInfo(
                f"Running {protocol} cross-validation over {n} observations "
                "(a full refit for every held-out point)..."
            )

            def _predict(train_coords, train_values, test_coords):
                local = rbf_weights(train_coords, train_values, kernel,
                                    shape_arg, smoothing)
                return rbf_predict(local, train_coords, test_coords)

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
            f"Evaluating a {cols} x {rows} grid ({cells:,} cells) at "
            f"{res_x:.6g} x {res_y:.6g} map units."
        )

        xs, ys = grid_centres(extent, cols, rows)
        gx, gy = np.meshgrid(xs, ys)
        targets = np.column_stack([gx.ravel(), gy.ravel()])

        # The surface is a single evaluation of the solved system, so the only
        # thing worth reporting progress on is the chunked distance matrix.
        prediction = np.empty(len(targets), dtype=float)
        for start, stop in iter_chunks(len(targets), n):
            if feedback.isCanceled():
                return {}
            prediction[start:stop] = rbf_predict(state, coords_arr,
                                                 targets[start:stop])
            feedback.setProgress(int(90.0 * stop / max(1, len(targets))))

        if feedback.isCanceled():
            return {}

        prediction_grid = prediction.reshape(rows, cols)

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

        feedback.pushInfo("Writing the predicted surface...")
        pred_path = self.parameterAsOutputLayer(parameters, self.PREDICTION, context)
        if not pred_path:
            raise QgsProcessingException(
                "No destination was given for the predicted surface."
            )
        try:
            write_float32_geotiff(pred_path, write_grid, extent,
                                  source.sourceCrs(), DEFAULT_NODATA,
                                  feedback=feedback)
        except (ValueError, RuntimeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

        stats = {
            "prediction": surface_stats(write_grid),
            "cols": cols,
            "rows": rows,
            "cells": cells,
            "res_x": float(res_x),
            "res_y": float(res_y),
        }

        feedback.setProgress(95)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, kernel, shape_arg, spacing, smoothing,
            state, cv, cv_skip_note, stats, numeric_summary, crs_warning, hull,
            mask_outside, max_points,
        )

        result = {
            self.PREDICTION: pred_path,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        feedback.setProgress(100)
        return result

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(
        self, path, field_name, n, kernel, shape_arg, spacing, smoothing,
        state, cv, cv_skip_note, stats, numeric_summary, crs_warning, hull,
        mask_outside, max_points,
    ):
        exact = state["max_abs_residual"] == 0.0
        verdict, color, verdict_text, next_action = validation_verdict(
            cv, numeric_summary["std"] or 0.0, "Radial basis interpolation"
        )
        if not exact:
            next_action += (
                f" The surface is not exact: smoothing of {smoothing:g} left a "
                f"largest residual of {state['max_abs_residual']:.6g} at the "
                "observations. If the intent was an exact interpolation, set "
                "smoothing to zero and compare the cross-validation scores."
            )

        pred_stats = stats["prediction"]
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
            ("Observation cap applied to this run", f"{max_points:,}"),
        ]

        if kernel == "thin_plate":
            shape_txt = "not applicable - the thin plate form is fixed"
        else:
            shape_txt = f"{shape_arg:.6g}"
            if shape_arg == spacing:
                shape_txt += " (derived from the observation spacing)"

        method_pairs = [
            ("Kernel", html.escape(kernel.replace("_", " "))),
            ("Kernel shape parameter (map units)", html.escape(shape_txt)),
            ("Smoothing", f"{smoothing:g}"),
            ("Surface is exact at the observations",
             "yes" if exact else "no"),
            ("RMS residual at the observations", fmt(state["rms_residual"])),
            ("Largest residual at the observations",
             fmt(state["max_abs_residual"])),
            ("Mean nearest-neighbour distance (map units)", f"{spacing:.6g}"),
        ]
        if state["augmented"]:
            method_pairs.append(
                ("Linear polynomial term", "present (required by this kernel)")
            )
        else:
            method_pairs.append(
                ("Linear polynomial term",
                 "absent - this kernel does not reproduce a plane exactly")
            )

        residual_note = (
            "The surface passes through every observation, so the residual is "
            "zero to rounding. That is the defining property of an exact "
            "interpolator and it is also why it says nothing about quality: a "
            "surface that is guaranteed to fit its own data must be judged on "
            "data it has not seen, which is what the cross-validation section "
            "does."
            if exact else
            f"Smoothing of {smoothing:g} was applied, so the surface no longer "
            f"passes through every observation: the RMS residual is "
            f"{state['rms_residual']:.6g} and the largest single residual is "
            f"{state['max_abs_residual']:.6g}, in the field's own units. That "
            "residual is what smoothing bought - resistance to noise in the "
            "observations - and it is reported here so the trade is visible "
            "rather than implied by a surface that looks smooth."
        )

        cv_note = cv_protocol_note(cv)
        if cv_skip_note:
            cv_note += " " + cv_skip_note
        cv_note += (
            " Every held-out point is scored from a system refitted without it, "
            "which is the only kind of score that means anything here: because "
            "removing an observation changes the solved system and not just its "
            "right-hand side, there is no closed form for this method's "
            "leave-one-out error."
        )

        extrapolation_html = (
            "<section>"
            "<h2>Outside the Observed Area</h2>"
            f"<p>{html.escape(hull_note(hull, mask_outside, DEFAULT_NODATA))}</p>"
            "</section>"
        )

        kernel_note = KERNEL_NOTES.get(kernel, "")

        guidance_html = analyst_guidance_html(
            "Radial Basis Function Interpolation (RBF)",
            "Fits a sum of radial basis functions through every observation, "
            "producing the smoothest surface of its kind that honours the data.",
            [
                "The kernel was chosen for the field's character and the choice "
                "confirmed by cross-validation, not by which map looks best.",
                "The shape parameter, where the kernel has one, is comparable "
                "to the spacing of the observations.",
                "The cross-validation error is well below the field's own "
                "standard deviation.",
            ],
            [
                "Reading the surface far outside the observations. The "
                "multiquadric kernel in particular grows with distance rather "
                "than flattening, so extrapolated values can leave the observed "
                "range entirely.",
                "Smoothing the surface to remove wiggles and then reporting it "
                "as an exact interpolation. Smoothing is a real trade and the "
                "residual it leaves is reported for that reason.",
                "Assuming a smooth surface is a correct one. Smoothness is "
                "built into the method; it is not evidence about the field.",
            ],
            [
                "Ordinary / Universal Kriging Interpolation on the same points, "
                "which reports an analytic error surface this method cannot",
                "Inverse Distance Weighted Interpolation for a faster baseline "
                "to compare the cross-validation score against",
                "Polynomial Trend Surface if the field varies smoothly at a "
                "scale larger than the observation spacing",
            ],
            "Radial basis interpolation answers where the value is, on the "
            "assumption that the surface should be as smooth as the data "
            "allows. That assumption is right for a field that is genuinely "
            "continuous and wrong for one with sharp boundaries - a geological "
            "contact, an administrative edge - where the method will smooth the "
            "discontinuity into a gradient that does not exist.",
        )

        caveat_text = [
            "Radial basis interpolation reproduces the observations exactly at "
            "zero smoothing, which means it also reproduces any noise in them. "
            "The surface is not filtered; whatever error is in the input "
            "appears in the output at the same locations.",
            "The method has no variance model, so the surface carries no "
            "per-cell uncertainty. The cross-validation score describes the "
            "method's average behaviour over the observations, not the "
            "reliability of any particular cell.",
            "The thin plate spline is the only kernel here with a polynomial "
            "term, and therefore the only one with linear precision. The other "
            "three do not reproduce a plane exactly, which is a mathematical "
            "property of the kernels rather than a defect of the "
            "implementation.",
            "A dense system is solved over all observations at once, so the "
            "result depends on every observation in the layer. Adding or "
            "removing a point changes the whole surface, not just its "
            "neighbourhood.",
        ]

        summary_txt = (
            f"{n} observations were fitted with the "
            f"{html.escape(kernel.replace('_', ' '))} kernel"
            + (f" and a shape parameter of {shape_arg:.6g} map units"
               if shape_arg is not None else "")
            + (f", with smoothing {smoothing:g}" if smoothing > 0
               else ", with no smoothing")
            + f". The surface is {'exact at the observations' if exact else 'no longer exact'} "
            "and carries no variance model. Cross-validation error was "
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
            "Radial Basis Function Interpolation (RBF)",
            f"Field Interpolated: <strong>{html.escape(field_name)}</strong> | "
            f"Observations: <strong>{n}</strong> | Kernel: "
            f"<strong>{html.escape(kernel.replace('_', ' '))}</strong>",
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
        <p>{html.escape(kernel_note)}</p>
        <p>{html.escape(residual_note)}</p>
    </section>

    <section>
        <h2>Cross-Validation</h2>
        <table>
            <thead><tr><th>Validation Statistic</th><th>Value</th></tr></thead>
            <tbody>{cv_metric_rows(cv)}</tbody>
        </table>
        <p>{html.escape(cv_note)}</p>
    </section>

    <section>
        <h2>Interpolated Surface</h2>
        <table>
            <thead><tr><th>Surface Property</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(surface_pairs)}</tbody>
        </table>
        <p>The surface was written as a GeoTIFF Float32 raster covering the
        extent of the input points; cells that could not be predicted carry the
        no-data value {DEFAULT_NODATA:g}.</p>
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
                caveats_html("Radial basis function interpolation", None,
                             numeric_summary, caveat_text),
                guidance_html,
            ))
