# -*- coding: utf-8 -*-
"""Natural Neighbour Interpolation Processing Algorithm."""
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
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsRectangle,
)

from ..core.interpolation_engines import (
    NATURAL_LOO_LIMIT,
    NATURAL_SOFT_LIMIT,
    cross_validate,
    cv_protocol_for,
    mean_nearest_distance,
    natural_neighbour,
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

#: Leave-one-out rebuilds the mesh once per held-out observation.  This is
#: measured, not estimated: the engine carries the timings beside the constant.
CV_LOO_LIMIT = NATURAL_LOO_LIMIT


class NaturalNeighbourAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    CELL_SIZE = "CELL_SIZE"
    MAX_POINTS = "MAX_POINTS"
    MASK_OUTSIDE = "MASK_OUTSIDE"
    RUN_CV = "RUN_CV"
    PREDICTION = "PREDICTION"
    NEIGHBOUR_COUNT = "NEIGHBOUR_COUNT"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "natural_neighbour_interpolation"

    def displayName(self) -> str:
        return "Natural Neighbour Interpolation (Sibson)"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("natural_neighbour_interpolation")

    def createInstance(self):
        return NaturalNeighbourAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Predicts a surface from scattered point observations by weighting "
            "each one by the share of the new observation's Voronoi cell it "
            "would give up to the new point - Sibson's natural-neighbour "
            "coordinates, which is what ArcGIS Pro's Natural Neighbor tool "
            "computes.\n\n"
            "The idea is local and geometric. Every observation owns a Voronoi "
            "cell: the area closer to it than to any other observation. Drop a "
            "new point into that arrangement and it takes a bite out of the "
            "cells around it. The size of the bite it takes from each neighbour "
            "is that neighbour's weight, normalised to sum to one. The surface "
            "that results is smooth, passes through every observation exactly, "
            "and - unlike inverse distance weighting or a multiquadric radial "
            "basis function - reproduces any plane through the data exactly.\n\n"
            "That last property is why this tool ships Sibson's coordinates "
            "rather than the cheaper Laplacian ones, which take Voronoi edge "
            "lengths instead of areas. Both are non-negative, both sum to one "
            "and both are continuous, so they look alike on a smooth field and "
            "are easy to confuse. The Laplacian form does not reproduce a "
            "plane; this was measured here rather than taken on trust, and on "
            "the engine's own test field the Laplacian variant missed a plane "
            "by about a tenth of the field's range while Sibson's matched it to "
            "1e-13. Natural-neighbour interpolation without linear precision is "
            "not what the name means, so the shipped weights are Sibson's.\n\n"
            "A target outside the observations' convex hull has no value here, "
            "and this is not a masking policy that can be switched off: the "
            "natural neighbours of a point are the vertices of the Voronoi cell "
            "it would have, and outside the hull that cell is unbounded, so "
            "there is nothing to compute. The hull option writes those cells as "
            "no-data, which is the only honest thing to write.\n\n"
            "The interpolant carries no variance model, so cross-validation is "
            "its only quality measure. Because removing an observation changes "
            f"the mesh itself, leave-one-out genuinely rebuilds it, which is "
            f"why above {NATURAL_LOO_LIMIT} observations it is replaced by "
            "k-fold. Those two numbers are not comparable and the report says "
            "which was used. One further consequence is worth knowing before "
            "reading the score: the observations on the convex hull cannot be "
            "predicted from the others, because they lie outside the reduced "
            "hull, so they are excluded from the statistics and counted "
            "separately in the report.\n\n"
            "Output: a prediction raster (GeoTIFF Float32 covering the extent "
            "of the input points), an optional raster of the number of natural "
            "neighbours behind each cell, and an HTML report with the "
            "cross-validation statistics and the share of the output the "
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
                "Maximum observations (the mesh build is close to quadratic in "
                "this number)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=NATURAL_SOFT_LIMIT,
                minValue=3,
                maxValue=200_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MASK_OUTSIDE,
                "Write cells outside the observations' convex hull as no-data "
                "(outside the hull this method has no value to write)",
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
                "Predicted surface",
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.NEIGHBOUR_COUNT,
                "Number of natural neighbours per cell (optional; a count, not "
                "an uncertainty)",
                optional=True,
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
                "HTML_REPORT_OUT", "Natural neighbour interpolation report"
            )
        )

    # ── run ──────────────────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
        mask_outside = self.parameterAsBool(parameters, self.MASK_OUTSIDE, context)
        run_cv = self.parameterAsBool(parameters, self.RUN_CV, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_natural_neighbour_report.html"
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
                f"Natural neighbour interpolation needs at least 3 observations "
                f"with a finite value and a usable geometry; got {n}."
            )
        require_non_constant(numeric_summary)
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. Every observation is inserted "
                "into the mesh one at a time from Python, so the build cost "
                "grows faster than the point count. Raise the cap deliberately "
                "if you accept the wait, or reduce the point count first."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        spacing = mean_nearest_distance(coords_arr)
        feedback.pushInfo(
            f"Mean nearest-neighbour distance: {spacing:.6g} map units."
        )

        if feedback.isCanceled():
            return {}

        # ── cross-validation, before the expensive part ──────────────────────
        cv = None
        cv_skip_note = None
        if run_cv:
            protocol, folds = cv_protocol_for(n, CV_LOO_LIMIT)
            if protocol == "kfold":
                cv_skip_note = (
                    f"Leave-one-out validation was not affordable at {n} "
                    f"observations, above the limit of {CV_LOO_LIMIT}. Each "
                    "held-out observation needs its own triangulation of all "
                    f"the others, so {folds}-fold validation was used instead: "
                    "it rebuilds the mesh once per fold rather than once per "
                    "observation. Its error is systematically larger than a "
                    "leave-one-out figure would have been, because each model "
                    "is fitted on less data, and the two must not be compared."
                )
                feedback.pushWarning(cv_skip_note)
            feedback.pushInfo(
                f"Running {protocol} cross-validation over {n} observations "
                "(each held-out point needs the mesh rebuilt without it)..."
            )

            def _predict(train_coords, train_values, test_coords):
                out, _, _ = natural_neighbour(train_coords, train_values,
                                              test_coords)
                return out

            try:
                cv = cross_validate(
                    _predict, coords_arr, values_arr,
                    protocol=protocol, folds=folds,
                )
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc
            feedback.pushInfo(
                f"Cross-validation ({cv['protocol']}): ME={cv['me']:.6g}, "
                f"RMSE={cv['rmse']:.6g}, MAE={cv['mae']:.6g}"
            )
            if cv["n_dropped"]:
                # These are the hull observations, and the count is expected to
                # be the hull size rather than a defect: a hull vertex lies
                # outside the hull of the points that remain once it is held
                # out, so it has no natural neighbours there and no prediction.
                feedback.pushInfo(
                    f"{cv['n_dropped']} of {n} observations could not be scored: "
                    "they lie on the convex hull, so once held out they fall "
                    "outside the hull of the rest and the method has no value "
                    "for them. They are excluded from the statistics below."
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
            f"{res_x:.6g} x {res_y:.6g} map units. Each cell is weighted "
            "individually, so a grid far larger than the default will take "
            "proportionally longer."
        )

        xs, ys = grid_centres(extent, cols, rows)
        gx, gy = np.meshgrid(xs, ys)
        targets = np.column_stack([gx.ravel(), gy.ravel()])

        def _progress(done, total):
            feedback.setProgress(int(90.0 * done / max(1, total)))
            return not feedback.isCanceled()

        try:
            prediction, used, detail = natural_neighbour(
                coords_arr, values_arr, targets, progress=_progress,
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except RuntimeError:
            return {}

        if feedback.isCanceled():
            return {}

        inside = np.isfinite(prediction)
        feedback.pushInfo(
            f"{int(inside.sum()):,} of {cells:,} cells were inside the "
            "observations' convex hull and were interpolated."
        )
        if detail["degenerate"]:
            # The fallback is a real answer with a different weighting, so the
            # count is reported rather than left to be discovered in the raster.
            feedback.pushWarning(
                f"{detail['degenerate']:,} cells had a numerically degenerate "
                "neighbour ring - the ring's Voronoi vertices coincided. Those "
                "cells were filled by inverse squared distance over the same "
                "ring, which keeps the surface bounded by the observations but "
                "is not the natural-neighbour value. They are counted here "
                "because the raster alone will not show them."
            )

        prediction_grid = prediction.reshape(rows, cols)

        hull = hull_summary(coords_arr, targets)
        if hull is not None and hull["outside"]:
            feedback.pushInfo(
                f"{hull['outside']:,} of {hull['total']:,} cells "
                f"({hull['fraction']:.1%}) fall outside the convex hull of the "
                "observation locations and carry no prediction."
            )

        write_grid = prediction_grid
        if mask_outside:
            write_grid = apply_hull_mask(prediction_grid, hull, DEFAULT_NODATA)
        else:
            feedback.pushWarning(
                "Cells outside the observation hull were left as no-data "
                "anyway: outside the hull a natural neighbour is undefined "
                "rather than merely unreliable, so this method has no value to "
                "write there."
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

        neighbour_path = None
        count_path = self.parameterAsOutputLayer(
            parameters, self.NEIGHBOUR_COUNT, context
        )
        if count_path:
            count_grid = used.reshape(rows, cols).astype(float)
            count_grid = np.where(np.isfinite(prediction_grid), count_grid,
                                  DEFAULT_NODATA)
            feedback.pushInfo("Writing the neighbour-count raster...")
            try:
                write_float32_geotiff(count_path, count_grid, extent,
                                      source.sourceCrs(), DEFAULT_NODATA,
                                      feedback=feedback)
            except (ValueError, RuntimeError) as exc:
                raise QgsProcessingException(str(exc)) from exc
            neighbour_path = count_path

        stats = {
            # Outside the hull this method writes the sentinel whatever the
            # mask option says, and the neighbour-count raster carries it in
            # the same cells, so both are summarised against it.
            "prediction": surface_stats(write_grid, DEFAULT_NODATA),
            "neighbour": (surface_stats(count_grid, DEFAULT_NODATA)
                          if neighbour_path else None),
            "cols": cols,
            "rows": rows,
            "cells": cells,
            "res_x": float(res_x),
            "res_y": float(res_y),
        }

        feedback.setProgress(95)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, spacing, cv, cv_skip_note, stats,
            numeric_summary, crs_warning, hull, mask_outside, max_points,
            detail,
        )

        result = {
            self.PREDICTION: pred_path,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        if neighbour_path:
            result[self.NEIGHBOUR_COUNT] = neighbour_path
        feedback.setProgress(100)
        return result

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(
        self, path, field_name, n, spacing, cv, cv_skip_note, stats,
        numeric_summary, crs_warning, hull, mask_outside, max_points, detail,
    ):
        verdict, color, verdict_text, next_action = validation_verdict(
            cv, numeric_summary["std"] or 0.0, "Natural neighbour interpolation"
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

        method_pairs = [
            ("Weighting scheme", "Sibson natural-neighbour coordinates"),
            ("Interpolant is exact at the observations", "yes, by construction"),
            ("Reproduces a plane exactly", "yes - the property linking Sibson's "
                                           "coordinates to the method's name"),
            ("Mean nearest-neighbour distance (map units)", f"{spacing:.6g}"),
            ("Cells that landed exactly on an observation",
             f"{detail['exact']:,}"),
            ("Cells with a degenerate neighbour ring",
             f"{detail['degenerate']:,}"),
            ("Cells inside the hull with no neighbour ring",
             f"{detail['no_ring']:,}"),
        ]

        exact_note = (
            "The surface passes through every observation exactly, which is "
            "what makes this an interpolator rather than a fit. The same "
            "property means the value at a datum is uninformative about "
            "quality: it is returned, not predicted. Only the cross-validation "
            "section says anything about how the method behaves where it has "
            "not been."
        )

        sibson_note = (
            "The weights are Sibson's: the area of the target's own Voronoi "
            "cell that each natural neighbour would surrender to it, "
            "normalised. A cheaper variant takes Voronoi edge lengths instead "
            "of areas - the Laplacian coordinates. Both are non-negative, both "
            "sum to one and both are continuous, so they are easily mistaken "
            "for one another, and only the Sibson form reproduces a plane "
            "exactly. The Laplacian form was implemented and measured against "
            "the same test during development and missed a plane by about a "
            "tenth of the field's range, so it is not what this tool ships."
        )

        degenerate_note = (
            "Every cell was weighted by Sibson's coordinates."
            if not detail["degenerate"] else
            f"{detail['degenerate']:,} cells had a neighbour ring whose Voronoi "
            "vertices coincided, which makes the coordinates degenerate. Those "
            "cells were filled by inverse squared distance over the same ring "
            "instead. That fallback keeps the surface bounded by the "
            "observations but it is not a natural-neighbour value, so those "
            "cells should be treated as no better than an inverse distance "
            "estimate."
        )

        cv_note = cv_protocol_note(cv)
        if cv_skip_note:
            cv_note += " " + cv_skip_note
        if cv is not None and cv["n_dropped"]:
            cv_note += (
                f" {cv['n_dropped']} of {n} observations are absent from these "
                "statistics. They lie on the convex hull, so once held out they "
                "fall outside the hull of the remaining observations and the "
                "method returns no value at all for them. This is a genuine "
                "limitation of natural-neighbour interpolation at the edge of a "
                "dataset, not a failure of the run, but it does mean the "
                "reported error describes the interior of the sampled area."
            )
        cv_note += (
            " Removing an observation from this method removes it from the "
            "mesh, so a held-out point is scored against a differently shaped "
            "triangulation - there is no closed form for the leave-one-out "
            "error here as there is for inverse distance weighting."
        )

        extrapolation_html = (
            "<section>"
            "<h2>Outside the Observed Area</h2>"
            f"<p>{html.escape(hull_note(hull, mask_outside, DEFAULT_NODATA))}</p>"
            "<p>For this method the hull is not a policy choice. A natural "
            "neighbour is a vertex of the Voronoi cell the target would have, "
            "and a target outside the hull has no such cell - its Voronoi "
            "region is unbounded in at least one direction. There is no "
            "neighbour set to weight, so there is nothing to compute and "
            "nothing to mask away.</p>"
            "</section>"
        )

        neighbour_note = (
            "A raster of neighbour counts was written beside the surface. It "
            "is a count of the observations that contributed to each cell - "
            "typically three to eight - and it is worth being plain that this "
            "is not an uncertainty. The method has no variance model, so no "
            "raster from this tool carries per-cell uncertainty, and a cell "
            "with eight neighbours is not thereby twice as reliable as one "
            "with four."
        )

        guidance_html = analyst_guidance_html(
            "Natural Neighbour Interpolation (Sibson)",
            "Weights each observation by the share of its Voronoi cell the "
            "target would capture, giving a smooth surface with linear "
            "precision that passes through every observation.",
            [
                "The cross-validation error is well below the field's own "
                "standard deviation.",
                "The observation density is roughly even, so no cell is "
                "weighted by a single remote neighbour.",
                "The area of interest lies inside the observations' convex "
                "hull, where the method is defined.",
            ],
            [
                "Reading the surface at or beyond the hull edge. There is no "
                "natural-neighbour value there at all, so a raster that shows "
                "one has been filled by something else.",
                "Treating the neighbour-count raster as a confidence surface. "
                "It counts contributors; it does not measure uncertainty.",
                "Assuming a smooth surface means a correct one. Smoothness is "
                "built into the method, not evidence about the field.",
            ],
            [
                "Ordinary / Universal Kriging Interpolation on the same points, "
                "which reports an analytic error surface this method cannot",
                "Radial Basis Function Interpolation with the thin plate "
                "spline, the other exact interpolator here with linear "
                "precision, for a cross-validation comparison",
                "Inverse Distance Weighted Interpolation for a faster baseline "
                "that will not reproduce a plane",
            ],
            "Natural-neighbour interpolation answers where the value is, on the "
            "assumption that the Voronoi geometry of the observation locations "
            "carries real information about the field. That assumption suits "
            "data whose sampling density varies, where each observation should "
            "speak for its own neighbourhood and no further, and it is wrong "
            "for a field whose variation has a scale and a direction that the "
            "sampling layout does not share.",
        )

        caveat_text = [
            "The method has no variance model, so the surface carries no "
            "per-cell uncertainty. The cross-validation score describes the "
            "method's average behaviour over the observations, not the "
            "reliability of any particular cell.",
            "Observations on the convex hull cannot be predicted from the "
            "others - once held out they lie outside the reduced hull - so the "
            "reported error describes the interior of the sampled area and says "
            "nothing about accuracy at its edge.",
            "The interpolant is defined only inside the convex hull of the "
            "observations. Outside it there is no value to write, so cells "
            "there are no-data regardless of how the mask option is set.",
            "The weights depend on the observation locations alone, not on the "
            "observed values, so an extreme value is spread over its "
            "neighbours' cells whether or not it represents a genuine local "
            "feature.",
        ]

        summary_txt = (
            f"{n} observations were interpolated with Sibson's natural-neighbour "
            "coordinates over a "
            f"{stats['cols']} x {stats['rows']} grid. The surface is exact at "
            "the observations and reproduces a plane exactly, and it carries no "
            "variance model. Cross-validation error was "
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
            "Natural Neighbour Interpolation (Sibson)",
            f"Field Interpolated: <strong>{html.escape(field_name)}</strong> | "
            f"Observations: <strong>{n}</strong> | Weighting: "
            "<strong>Sibson coordinates</strong>",
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
        <p>{html.escape(sibson_note)}</p>
        <p>{html.escape(exact_note)}</p>
        <p>{html.escape(degenerate_note)}</p>
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
        <p>{html.escape(neighbour_note)}</p>
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
                caveats_html("Natural neighbour interpolation", None,
                             numeric_summary, caveat_text),
                guidance_html,
            ))
