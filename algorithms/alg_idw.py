# -*- coding: utf-8 -*-
"""Inverse Distance Weighted Interpolation Processing Algorithm."""
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
    IDW_SOFT_LIMIT,
    NEIGHBOURHOODS,
    cross_validate,
    cv_protocol_for,
    idw,
    mean_nearest_distance,
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

NEIGHBOURHOOD_LABELS = (
    "All observations",
    "Nearest N observations",
    "All observations within a radius",
)

#: How many automatic radii to allow in the search-neighbourhood mode.  Three
#: mean nearest-neighbour distances is a conventional starting point: it
#: reaches far enough to pick up the second ring of observations in a roughly
#: regular pattern, which is the smallest neighbourhood that shows any
#: structure beyond the single closest point.
AUTO_RADIUS_FACTOR = 3.0

#: Leave-one-out is a genuine refit here, so it costs one pass over the
#: observations per observation, and the total is quadratic in the count.
#: Above this it is replaced by k-fold rather than skipped, because a k-fold
#: figure that is honestly labelled beats no figure at all.
CV_LOO_LIMIT = 2000


class IdwAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    POWER = "POWER"
    NEIGHBOURHOOD = "NEIGHBOURHOOD"
    NEIGHBOURS = "NEIGHBOURS"
    RADIUS = "RADIUS"
    CELL_SIZE = "CELL_SIZE"
    MAX_POINTS = "MAX_POINTS"
    MASK_OUTSIDE = "MASK_OUTSIDE"
    RUN_CV = "RUN_CV"
    PREDICTION = "PREDICTION"
    NEIGHBOUR_COUNT = "NEIGHBOUR_COUNT"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "idw_interpolation"

    def displayName(self) -> str:
        return "Inverse Distance Weighted Interpolation (IDW)"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("idw_interpolation")

    def createInstance(self):
        return IdwAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Predicts a continuous surface from scattered point observations by "
            "inverse distance weighting: each cell is a weighted average of the "
            "observations, weighted by one over distance raised to a power.\n\n"
            "Inverse distance weighting is the simplest interpolator that "
            "deserves the name. It assumes nothing about the field beyond "
            "Tobler's first law - things near each other are more alike - and it "
            "has no parameters to fit and no model to validate. That is its "
            "strength and its whole problem. It will always produce a surface; "
            "it has no way of telling you whether that surface means anything. "
            "A cell between two clusters of observations is filled in with the "
            "average of the two clusters' values, whether or not anything "
            "happens there.\n\n"
            "The power is the entire method. At 1 the surface is close to a "
            "nearest-neighbour map; 2 is the conventional default; past about 4 "
            "the nearest few observations dominate so completely that the "
            "others stop mattering and you get the isolated bull's-eyes the "
            "method is criticised for. If a change in power visibly changes the "
            "map, the map is being made by the parameter and not by the data.\n\n"
            "Because the weights depend only on geometry, the interpolation is "
            "exact: a cell landing on an observation takes that observation's "
            "value. It cannot extrapolate, because a weighted average can only "
            "fall between the values that went into it - a property that makes "
            "the surface look safe outside the sampled area when it is really "
            "just uninformative there.\n\n"
            "With no variance model behind the weights, cross-validation is the "
            "only measure of quality this tool can offer. Every observation is "
            "held out in turn and predicted from the others; the report gives "
            "the resulting error against the field's own spread, so you can see "
            "whether the surface beats simply predicting the mean everywhere. "
            "Above 2000 observations leave-one-out becomes unaffordable and is "
            "replaced by k-fold, and the report says which was used, because "
            "the two are not comparable.\n\n"
            "The neighbour count raster is the nearest thing here to an "
            "uncertainty map, and it is not one. It shows how many observations "
            "each cell was averaged from. A cell built from two points and a "
            "cell built from twenty can carry the same number, and the first is "
            "far less trustworthy - but the count is at least an honest "
            "statement of what went into the cell, which a variance would not "
            "be, since this method has none to report.\n\n"
            "Output: a prediction raster (GeoTIFF Float32 covering the extent "
            "of the input points), an optional neighbour-count raster, and an "
            "HTML report with the cross-validation statistics, the search "
            "neighbourhood actually used, and how much of the output the "
            "observations do not span."
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
            QgsProcessingParameterNumber(
                self.POWER,
                "Distance weighting power",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.1,
                maxValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.NEIGHBOURHOOD,
                "Observations used for each cell",
                options=list(NEIGHBOURHOOD_LABELS),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.NEIGHBOURS,
                "Neighbour count (nearest-neighbour mode only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=12,
                minValue=1,
                maxValue=100_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS,
                "Search radius in map units (radius mode only; 0 = three times "
                "the mean nearest-neighbour distance)",
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
                "Maximum observations to interpolate from",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=IDW_SOFT_LIMIT,
                minValue=2,
                maxValue=2_000_000,
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
            QgsProcessingParameterRasterDestination(
                self.NEIGHBOUR_COUNT,
                "Neighbour count per cell (not an error surface)",
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
            QgsProcessingOutputHtml("HTML_REPORT_OUT", "IDW interpolation report")
        )

    # ── run ──────────────────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        power = self.parameterAsDouble(parameters, self.POWER, context)
        mode = NEIGHBOURHOODS[
            self.parameterAsEnum(parameters, self.NEIGHBOURHOOD, context)
        ]
        k = self.parameterAsInt(parameters, self.NEIGHBOURS, context)
        radius = self.parameterAsDouble(parameters, self.RADIUS, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
        mask_outside = self.parameterAsBool(parameters, self.MASK_OUTSIDE, context)
        run_cv = self.parameterAsBool(parameters, self.RUN_CV, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_idw_report.html"
            )

        feedback.pushInfo("Reading point locations and target values...")
        read = read_point_values(source, field_name, feedback)
        if read is None:
            return {}
        coords, values, numeric_summary, crs_warning = read
        push_diagnostics(feedback, numeric_summary, None, crs_warning)

        n = len(values)
        if n < 2:
            raise QgsProcessingException(
                f"Inverse distance weighting needs at least 2 observations with "
                f"a finite value and a usable geometry; got {n}."
            )
        require_non_constant(numeric_summary)
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. Raise the cap deliberately if "
                "you accept the wait, or reduce the point count first."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        spacing = mean_nearest_distance(coords_arr)
        radius_arg = None
        if mode == "radius":
            radius_arg = radius if radius > 0 else AUTO_RADIUS_FACTOR * spacing
            if radius > 0:
                feedback.pushInfo(f"Search radius: {radius_arg:.6g} map units.")
            else:
                feedback.pushInfo(
                    f"Search radius defaulted to {radius_arg:.6g} map units, "
                    f"three times the mean nearest-neighbour distance of "
                    f"{spacing:.6g}."
                )
        k_arg = k if mode == "nearest" else None
        if mode == "nearest":
            feedback.pushInfo(f"Using the {k} nearest observation(s) per cell.")

        if mode == "radius" and radius_arg < spacing:
            feedback.pushWarning(
                f"The search radius ({radius_arg:.6g}) is smaller than the mean "
                f"distance between neighbouring observations ({spacing:.6g}), so "
                "some cells will have no observation in range and will be "
                "written as no-data. That is the honest result rather than a "
                "fault, but widen the radius if you meant to fill the grid."
            )

        # ── cross-validation, before the expensive part ──────────────────────
        cv = None
        cv_skip_note = None
        if run_cv:
            protocol, folds = cv_protocol_for(n, CV_LOO_LIMIT)
            feedback.pushInfo(
                f"Running {protocol} cross-validation over {n} observations "
                "(each held-out point is predicted from a fresh fit)..."
            )

            def _predict(train_coords, train_values, test_coords):
                out, _used, _hits = idw(
                    train_coords, train_values, test_coords, power, k_arg,
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
                    f"{cv['n_dropped']} held-out observation(s) had no "
                    "observation left in range and were excluded from the "
                    "statistics."
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
            f"Interpolating a {cols} x {rows} grid ({cells:,} cells) at "
            f"{res_x:.6g} x {res_y:.6g} map units from {n} observations with "
            f"power {power:g}."
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
            prediction, used, exact_hits = idw(
                coords_arr, values_arr, targets, power, k_arg, radius_arg,
                progress=_progress,
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except RuntimeError:
            return {}

        if feedback.isCanceled():
            return {}

        prediction_grid = prediction.reshape(rows, cols)
        used_grid = used.reshape(rows, cols).astype(float)

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

        count_path = self.parameterAsOutputLayer(
            parameters, self.NEIGHBOUR_COUNT, context
        )
        if count_path:
            feedback.pushInfo("Writing the neighbour count surface...")
            try:
                write_float32_geotiff(count_path, used_grid, extent,
                                      source.sourceCrs(), DEFAULT_NODATA,
                                      feedback=feedback)
            except (ValueError, RuntimeError) as exc:
                raise QgsProcessingException(str(exc)) from exc

        stats = {
            "prediction": surface_stats(write_grid),
            "used": surface_stats(used_grid),
            "cols": cols,
            "rows": rows,
            "cells": cells,
            "res_x": float(res_x),
            "res_y": float(res_y),
        }

        feedback.setProgress(95)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, power, mode, k_arg, radius_arg, spacing,
            cv, cv_skip_note, stats, numeric_summary, crs_warning, hull,
            mask_outside, count_path, exact_hits, max_points,
        )

        result = {
            self.PREDICTION: pred_path,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        if count_path:
            result[self.NEIGHBOUR_COUNT] = count_path
        feedback.setProgress(100)
        return result

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(
        self, path, field_name, n, power, mode, k_arg, radius_arg, spacing,
        cv, cv_skip_note, stats, numeric_summary, crs_warning, hull,
        mask_outside, count_path, exact_hits, max_points,
    ):
        verdict, color, verdict_text, next_action = validation_verdict(
            cv, numeric_summary["std"] or 0.0, "Inverse distance weighting"
        )
        if cv is not None and cv["n_scored"] < n:
            next_action += (
                f" {n - cv['n_scored']} observation(s) could not be scored "
                "because their neighbourhood emptied when they were held out, "
                "which is itself a statement about the search radius."
            )

        if mode == "nearest":
            neighbourhood_txt = f"the {k_arg} nearest observation(s)"
        elif mode == "radius":
            neighbourhood_txt = (
                f"every observation within {radius_arg:.6g} map units"
            )
        else:
            neighbourhood_txt = "every observation in the layer"

        pred_stats = stats["prediction"]
        used_stats = stats["used"]
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
            ("Neighbour count, minimum", f"{used_stats['min']:.0f}"),
            ("Neighbour count, mean", f"{used_stats['mean']:.2f}"),
            ("Cells landing exactly on an observation", f"{exact_hits:,}"),
            ("Observation cap applied to this run", f"{max_points:,}"),
        ]

        method_pairs = [
            ("Distance weighting power", f"{power:g}"),
            ("Observations used per cell", html.escape(neighbourhood_txt)),
            ("Mean nearest-neighbour distance (map units)", f"{spacing:.6g}"),
        ]
        if mode == "radius":
            method_pairs.append(
                ("Search radius (map units)", f"{radius_arg:.6g}")
            )

        neighbour_note = (
            "Each cell was averaged from the observations listed above, so "
            "this raster counts what went into each prediction. It is not an "
            "error surface and must not be read as one: a cell averaged from "
            "two observations is not twice as uncertain as a cell averaged "
            "from four, because the method provides no way to say how uncertain "
            "either is. Its use is diagnostic - where the count dips, the "
            "surface is being carried by few observations and a change in the "
            "search neighbourhood would move it."
        )
        if not count_path:
            neighbour_note = (
                "No destination was given for the neighbour count, so the "
                "counts behind each cell are summarised above but not written "
                "as a raster. The summary still says how thin the support is "
                "over the grid as a whole."
            )

        cv_note = cv_protocol_note(cv)
        if cv_skip_note:
            cv_note += " " + cv_skip_note
        cv_note += (
            " Because inverse distance weighting is exact at the observations, "
            "the residuals against the fitted surface are all zero and would "
            "have said nothing; every number above comes from a refit that did "
            "not see the point being predicted."
        )

        extrapolation_html = (
            "<section>"
            "<h2>Outside the Observed Area</h2>"
            f"<p>{html.escape(hull_note(hull, mask_outside, DEFAULT_NODATA))}</p>"
            "</section>"
        )

        guidance_html = analyst_guidance_html(
            "Inverse Distance Weighted Interpolation (IDW)",
            "Predicts a continuous surface as a distance-weighted average of "
            "the observations, with no fitted model and no variance.",
            [
                "The power is set from knowledge of the field, or chosen by "
                "comparing cross-validation scores rather than by how the map "
                "looks.",
                "The search neighbourhood is wide enough that no cell is built "
                "from a single observation.",
                "The cross-validation error is well below the field's own "
                "standard deviation, so the surface carries real spatial "
                "information.",
            ],
            [
                "Reading the surface between clusters of observations. The "
                "method fills those gaps with the average of the surrounding "
                "values and has no way to know that anything changes there.",
                "Raising the power to make the map look sharper. Past about 4 "
                "the bull's-eyes around individual observations are an artefact "
                "of the parameter, not of the field.",
                "Treating a smooth surface outside the sampled area as "
                "meaningful. It is a weighted average that cannot leave the "
                "observed range, which looks safe and is merely uninformative.",
            ],
            [
                "Ordinary / Universal Kriging Interpolation on the same points, "
                "for a surface that carries its own error estimate",
                "Radial Basis Function interpolation for a smoother surface "
                "that reproduces a trend exactly",
                "Local Moran's I to check whether the pattern in the "
                "observations is real before trusting the pattern in the map",
            ],
            "Inverse distance weighting answers where the value is, on the "
            "assumption that near things are alike and that this is all there "
            "is to know. It is a reasonable default when nothing is known about "
            "the field's structure, and the wrong tool once something is - the "
            "moment the cross-validation error stops improving with a wider "
            "neighbourhood, the data has structure that a variogram would "
            "describe and this method cannot.",
        )

        caveat_text = [
            "Inverse distance weighting assumes the value at a location is a "
            "weighted average of nearby values and nothing more. It has no "
            "model of the field's spatial structure, so it cannot show that "
            "this assumption holds, and the surface it produces is not evidence "
            "that it does.",
            "The method is exact at the observations and smooth everywhere "
            "else, which produces local extrema at the data points and flat "
            "regions between them. Both are artefacts of the weighting, not "
            "features of the field.",
            "A weighted average can never fall outside the range of the values "
            "that went into it, so the surface cannot show a peak or a trough "
            "that was not sampled. An extreme value between observations is "
            "invisible to this method by construction.",
            "Predictions outside the convex hull of the observations are "
            "extrapolation in name only - the numbers there are still averages "
            "of real observations, but nothing constrains how far the true "
            "field differs from them.",
        ]

        summary_txt = (
            f"{n} observations were averaged with a distance power of {power:g}, "
            f"each cell using {html.escape(neighbourhood_txt)}. The surface is "
            f"exact at the observations and cannot leave the range of the "
            f"observed values. Cross-validation error was "
            f"<strong>{'not measured' if cv is None else fmt(cv['rmse'], 6)}</strong>"
            + (
                f", which is {cv['nrmse']:.3f} of the field's standard deviation "
                f"of {numeric_summary['std']:.6g}"
                if cv is not None and cv.get("nrmse") is not None
                else ""
            )
            + ". This method carries no variance model, so nothing below reports "
            "a per-cell uncertainty - the cross-validation score is the whole "
            "of the evidence about how well it predicts."
        )

        head = report_head(
            "Inverse Distance Weighted Interpolation (IDW)",
            f"Field Interpolated: <strong>{html.escape(field_name)}</strong> | "
            f"Observations: <strong>{n}</strong> | Power: "
            f"<strong>{power:g}</strong> | Neighbourhood: "
            f"<strong>{html.escape(neighbourhood_txt)}</strong>",
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
        <p>The power is the whole of the method: the weight given to an
        observation falls as one over its distance raised to this number.
        Cross-validation is what should decide it. If raising the power makes
        the map look sharper without lowering the cross-validation error, the
        extra sharpness is being invented by the parameter.</p>
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
        <div class="variance-box"><p style="margin:0;">{html.escape(neighbour_note)}</p></div>
        <p style="margin-top:14px;">The surface and the neighbour counts were
        written as GeoTIFF Float32 rasters covering the extent of the input
        points; cells that could not be predicted carry the no-data value
        {DEFAULT_NODATA:g}.</p>
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
                caveats_html("Inverse distance weighted interpolation", None,
                             numeric_summary, caveat_text),
                guidance_html,
            ))
