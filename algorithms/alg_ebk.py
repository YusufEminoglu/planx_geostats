# -*- coding: utf-8 -*-
"""Empirical Bayesian Kriging Interpolation Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsRectangle,
    QgsWkbTypes,
)
from qgis.PyQt.QtCore import QVariant

from ..core.ebk_engine import (
    EBK_CV_FOLDS,
    EBK_CV_LIMIT,
    EBK_DEFAULT_SIMULATIONS,
    EBK_DEFAULT_SUBSET,
    EBK_MAX_SIMULATIONS,
    EBK_MIN_FITS,
    EBK_MIN_POINTS,
    EBK_MODELS,
    EBK_SOFT_LIMIT,
    check_workload,
    ebk_fold_cross_validation,
    ebk_predict,
    spatial_subsets,
)
from ..core.geostat_engines import DRIFTS, quality_flags
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
from ..core.reporting import analyst_guidance_html

from ._icons import algorithm_icon
from ._input import read_point_values, require_non_constant


logger = logging.getLogger("PlanX GeoStats Lab")

#: The tool's model enum.  "Automatic" comes first because the whole point of
#: EBK is that a user who does not know which variogram family fits their field
#: does not have to find out - which is the reader this tool is written for.
#: The remaining labels are generated from the engine's tuple so that the two
#: cannot drift apart, and the values are kept beside the labels for the same
#: reason.
EBK_MODEL_VALUES = ("auto",) + tuple(EBK_MODELS)
EBK_MODEL_LABELS = ("Automatic - whichever family fits best",) + tuple(
    name.capitalize() for name in EBK_MODELS
)

DRIFT_LABELS = (
    "None",
    "Linear trend",
    "Quadratic trend",
)

#: A fold split needs enough observations that each fold still has a variogram
#: to fit; the engine refuses below twice its minimum, so the tool checks the
#: same floor rather than discovering it from an exception.
CV_MIN_OBSERVATIONS = EBK_MIN_POINTS * 2

#: Attribute names of the per-subset diagnostics layer.
SUBSET_LAYER_FIELDS = (
    "subset", "family", "npoints", "nugget", "psill", "range", "r2",
    "effective_models", "max_weight", "target_cells", "fallback",
)

_FIELD_TYPES = {
    "subset": (QVariant.Int, 0, 0),
    "family": (QVariant.String, 20, 0),
    "npoints": (QVariant.Int, 0, 0),
    "nugget": (QVariant.Double, 20, 8),
    "psill": (QVariant.Double, 20, 8),
    "range": (QVariant.Double, 20, 8),
    "r2": (QVariant.Double, 20, 8),
    "effective_models": (QVariant.Double, 20, 4),
    "max_weight": (QVariant.Double, 20, 6),
    "target_cells": (QVariant.Int, 0, 0),
    "fallback": (QVariant.Int, 0, 0),
}


def subset_output_fields() -> QgsFields:
    """Build the diagnostics layer's fields from the name contract above."""
    fields = QgsFields()
    for name in SUBSET_LAYER_FIELDS:
        variant, length, precision = _FIELD_TYPES[name]
        if variant == QVariant.Int:
            fields.append(QgsField(name, variant))
        else:
            fields.append(QgsField(name, variant, len=length, prec=precision))
    return fields


def _median_of(items, key):
    """The median of one parameter across a list of dicts, or None.

    A median rather than a mean because the automatic fit can settle on a very
    different family in a handful of subsets, and one such outlier would drag a
    mean to somewhere no subset actually is.
    """
    values = []
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            values.append(value)
    return float(np.median(values)) if values else None


def _typical_fit(detail):
    """A median fitted variogram across every subset, for the quality rules.

    ``quality_flags`` judges a *single* fitted variogram, and this method does
    not have one - it has one per subset.  The median is the stand-in, and it
    is fed to those rules with a leave-one-out argument of ``None`` because the
    calibration flags would otherwise be worded for a protocol this tool does
    not use.

    Every key the rules read is present and numeric, defaulting to zero, so a
    parameter no subset produced suppresses a rule instead of tripping it.
    ``r2`` is the exception and is omitted entirely when absent, because the
    rule reading it treats zero as a real value.

    Returns ``(fit, median_max_lag)``.
    """
    fits = [s["initial"] for s in detail["subsets"] if s.get("initial")]
    fit = {"nugget": 0.0, "psill": 0.0, "sill": 0.0, "range": 0.0}
    nugget = _median_of(fits, "nugget")
    psill = _median_of(fits, "psill")
    if nugget is not None and psill is not None:
        fit["nugget"] = nugget
        fit["psill"] = psill
        fit["sill"] = nugget + psill
    range_ = _median_of(fits, "range")
    if range_ is not None:
        fit["range"] = range_
    r2 = _median_of(fits, "r2")
    if r2 is not None:
        fit["r2"] = r2
    return fit, _median_of(detail["subsets"], "max_lag")


class EmpiricalBayesianKrigingAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    MODEL = "MODEL"
    ALPHA = "ALPHA"
    DRIFT = "DRIFT"
    N_SIMS = "N_SIMS"
    SUBSET_SIZE = "SUBSET_SIZE"
    OVERLAP = "OVERLAP"
    SEED = "SEED"
    N_BINS = "N_BINS"
    CELL_SIZE = "CELL_SIZE"
    MAX_POINTS = "MAX_POINTS"
    RUN_CV = "RUN_CV"
    CV_FOLDS = "CV_FOLDS"
    MASK_OUTSIDE = "MASK_OUTSIDE"
    PREDICTION = "PREDICTION"
    VARIANCE = "VARIANCE"
    SUBSET_LAYER = "SUBSET_LAYER"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "ebk_interpolation"

    def displayName(self) -> str:
        return "Empirical Bayesian Kriging Interpolation"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("ebk_interpolation")

    def createInstance(self):
        return EmpiricalBayesianKrigingAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Predicts a continuous surface from scattered point observations by "
            "Empirical Bayesian Kriging, fitting the variogram for you, and "
            "writes a prediction variance that includes the uncertainty in the "
            "variogram itself.\n\n"
            "Ordinary kriging asks you to choose a variogram model and treats "
            "whatever you chose as exactly true. Empirical Bayesian Kriging does "
            "not. It splits the observations into local subsets, fits a "
            "variogram in each, simulates a new dataset from that fit, refits to "
            "the simulated data, and keeps every refit weighted by how well it "
            "explains the real data. The prediction is the weighted average over "
            "all those models, and the variance is their spread plus the "
            "ordinary kriging variance of each - which is the part a "
            "single-variogram kriging cannot report, because it has only one "
            "model and can only be wrong about it. Measured on synthetic fields "
            "with a known structure, that makes the reported variance about 8 "
            "per cent wider on average, and about 27 per cent wider close to the "
            "observations.\n\n"
            "This is the interpolator to reach for when you do not already know "
            "the field's variogram. Leave the model on Automatic: choosing the "
            "wrong family turns out to cost about one per cent of prediction "
            "error on a test field, so the automatic choice is not a compromise. "
            "It is also the tool to use when the reported uncertainty matters "
            "more than the last decimal of the prediction: no manual variogram "
            "fitting is involved, and none is expected of you.\n\n"
            "Two things about the variance surface will look wrong and are not. "
            "It is inflated where the observations are dense rather than where "
            "they are sparse, because that is where the fitted subsets disagree "
            "most about local structure. And it never falls below the ordinary "
            "kriging variance for the same model - it is that variance plus a "
            "non-negative term, so a directly comparable kriging run will always "
            "look more confident than this one.\n\n"
            "The cost is not the cubic solve that makes ordinary kriging "
            "expensive; it is the number of output cells multiplied by the "
            "number of simulations multiplied by the observations each "
            "simulation solves over. That product is bounded here rather than "
            "left to run: a configuration needing more than about three minutes "
            "is refused with the three knobs named, and one needing more than "
            "about twenty seconds is run with a warning. Raising the cell size, "
            "lowering the simulations or lowering the subset size are the three "
            "ways down.\n\n"
            "Validation is five-fold cross-validation, not leave-one-out. A "
            "held-out point costs this method a complete refit rather than one "
            "solve, so leave-one-out is out of reach here, and the report states "
            "the protocol because a five-fold figure is not comparable with a "
            "leave-one-out figure from another tool.\n\n"
            "Output: a prediction raster and a variance raster, both GeoTIFF "
            "Float32 covering the extent of the input points; an optional point "
            "layer carrying each subset's fitted family and parameters, so the "
            "auto-fitting can be inspected rather than trusted; and an HTML "
            "report."
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
                self.MODEL,
                "Semivariogram family (Automatic fits every subset to all of "
                "them and keeps the best)",
                options=list(EBK_MODEL_LABELS),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ALPHA,
                "Stable model shape exponent (Stable only; ignored otherwise)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.5,
                minValue=0.1,
                maxValue=2.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DRIFT,
                "Trend removed before fitting",
                options=list(DRIFT_LABELS),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_SIMS,
                "Simulations per subset (cost is linear in this number; below "
                "about 20 the average is visibly noisy)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=EBK_DEFAULT_SIMULATIONS,
                minValue=1,
                maxValue=EBK_MAX_SIMULATIONS,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SUBSET_SIZE,
                "Observations per local subset (0 = one subset holding every "
                "observation; the main cost knob)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=EBK_DEFAULT_SUBSET,
                minValue=0,
                maxValue=200_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.OVERLAP,
                "Rings of neighbouring subsets added to each subset",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=0,
                maxValue=5,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SEED,
                "Random seed (the same seed reproduces the same surface "
                "exactly)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0,
                minValue=0,
                maxValue=2_147_483_647,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_BINS,
                "Number of distance bins for each internal fit",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=15,
                minValue=4,
                maxValue=200,
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
                "Maximum observations this run will accept",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=EBK_SOFT_LIMIT,
                minValue=EBK_MIN_POINTS,
                maxValue=200_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_CV,
                "Run five-fold cross-validation (the only quality measure this "
                "method has)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CV_FOLDS,
                "Cross-validation folds",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=EBK_CV_FOLDS,
                minValue=2,
                maxValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MASK_OUTSIDE,
                "Write cells outside the observations' convex hull as no-data "
                "(this method will return a number there, but nothing "
                "constrains it)",
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
                self.VARIANCE,
                "Prediction variance surface (includes the variogram "
                "uncertainty that ordinary kriging omits)",
                optional=True,
                createByDefault=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.SUBSET_LAYER,
                "Per-subset fitted models (optional; one point per subset)",
                QgsProcessing.TypeVectorPoint,
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
                "HTML_REPORT_OUT", "Empirical Bayesian Kriging report"
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
        model = EBK_MODEL_VALUES[self.parameterAsEnum(parameters, self.MODEL, context)]
        alpha = self.parameterAsDouble(parameters, self.ALPHA, context)
        drift = DRIFTS[self.parameterAsEnum(parameters, self.DRIFT, context)]
        n_sims = self.parameterAsInt(parameters, self.N_SIMS, context)
        subset_size = self.parameterAsInt(parameters, self.SUBSET_SIZE, context)
        overlap = self.parameterAsInt(parameters, self.OVERLAP, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        n_bins = self.parameterAsInt(parameters, self.N_BINS, context)
        run_cv = self.parameterAsBool(parameters, self.RUN_CV, context)
        folds = self.parameterAsInt(parameters, self.CV_FOLDS, context)
        mask_outside = self.parameterAsBool(
            parameters, self.MASK_OUTSIDE, context
        )

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_ebk_report.html"
            )

        feedback.pushInfo("Reading point locations and target values...")
        read = read_point_values(source, field_name, feedback)
        if read is None:
            return {}
        coords, values, numeric_summary, crs_warning = read
        push_diagnostics(feedback, numeric_summary, None, crs_warning)

        n = len(values)
        if n < EBK_MIN_POINTS:
            raise QgsProcessingException(
                f"Empirical Bayesian Kriging needs at least {EBK_MIN_POINTS} "
                f"observations with a finite value and a usable geometry, so "
                f"that each subset has a variogram to fit; got {n}."
            )
        require_non_constant(numeric_summary)
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. Every simulation fits and "
                "factorises a covariance matrix of the subset's size, so the "
                "cost grows with the observation count. Raise the cap "
                "deliberately if you accept the wait, or reduce the number of "
                "points first - lowering the subset size is the cheaper way to "
                "make a large layer affordable."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)

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

        xs, ys = grid_centres(extent, cols, rows)
        gx, gy = np.meshgrid(xs, ys)
        targets = np.column_stack([gx.ravel(), gy.ravel()])

        hull = hull_summary(coords_arr, targets)

        # The workload is checked here - before cross-validation, not after it -
        # against the grid that will actually be evaluated and the subsets that
        # will actually be fitted.  An unaffordable configuration is refused
        # then and there rather than after minutes of validation work.
        try:
            subset_info = spatial_subsets(coords_arr, subset_size, overlap)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        mean_local = (float(np.mean(subset_info["sizes"])) if subset_info["sizes"]
                      else float(n))
        try:
            work_note = check_workload(cells, n_sims, mean_local)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        feedback.pushInfo(
            f"Evaluating a {cols} x {rows} grid ({cells:,} cells) from {n} "
            f"observations in {len(subset_info['subsets'])} subsets, "
            f"{n_sims} simulations each."
        )
        if work_note:
            feedback.pushWarning("This run is large: " + work_note)

        # ── cross-validation, before the expensive part ──────────────────────
        cv = None
        cv_skip_note = None
        if run_cv:
            if n > EBK_CV_LIMIT:
                cv_skip_note = (
                    f"Cross-validation was skipped: each fold refits and "
                    f"re-simulates the whole model in every subset, and this "
                    f"layer has {n} observations, above the limit of "
                    f"{EBK_CV_LIMIT}. The surface is still produced and still "
                    "carries a variance, but nothing in this report tests "
                    "whether that variance is calibrated. Lower the observation "
                    "count, or lower the subset size and simulations first."
                )
                feedback.pushWarning(cv_skip_note)
            elif n < CV_MIN_OBSERVATIONS:
                cv_skip_note = (
                    f"Cross-validation was skipped: a {folds}-fold split of "
                    f"{n} observations would leave fewer than {EBK_MIN_POINTS} "
                    "to fit a variogram on, and this method has nothing to fit "
                    "without them. Add observations, or lower the fold count."
                )
                feedback.pushWarning(cv_skip_note)
            else:
                feedback.pushInfo(
                    f"Running {folds}-fold cross-validation over {n} "
                    f"observations with {n_sims} simulations and subset size "
                    f"{subset_size} (each fold refits every subset)..."
                )
                try:
                    cv = ebk_fold_cross_validation(
                        coords_arr, values_arr, model, alpha, n_sims,
                        subset_size, overlap, folds=folds, seed=seed,
                        n_bins=n_bins, drift=drift,
                    )
                except ValueError as exc:
                    raise QgsProcessingException(str(exc)) from exc
                feedback.pushInfo(
                    f"Cross-validation ({cv['protocol']}): ME={cv['me']:.6g}, "
                    f"RMSE={cv['rmse']:.6g}, MSE={cv['mse']:.6g}, "
                    f"RMSSE={cv['rmsse']:.6g}"
                )
                if feedback.isCanceled():
                    return {}

        def _progress(done, total):
            feedback.setProgress(int(90.0 * done / max(1, total)))
            return not feedback.isCanceled()

        try:
            prediction, variance, detail = ebk_predict(
                coords_arr, values_arr, targets, model, alpha, n_sims,
                subset_size, overlap, seed=seed, n_bins=n_bins, drift=drift,
                progress=_progress,
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except RuntimeError:
            return {}

        if feedback.isCanceled():
            return {}

        feedback.pushInfo(
            f"Fitted family per subset: {', '.join(detail['families'])}. "
            f"Effective model count "
            f"{detail['effective_models']:.2f} of {n_sims} simulations."
        )
        if detail["fallbacks"]:
            feedback.pushWarning(
                f"{detail['fallbacks']} of {detail['n_subsets']} subsets kept "
                f"fewer than {EBK_MIN_FITS} successfully refitted models, so "
                "those subsets fell back to the single initial fit. Their "
                "predictions are ordinary kriging under one variogram and their "
                "variance omits the model-uncertainty term. The report says so "
                "as well; lowering the subset size usually removes the cause."
            )
        if detail["n_failed"]:
            feedback.pushInfo(
                f"{detail['n_failed']} simulations were discarded because the "
                "refit to the simulated data did not converge. They are a small "
                "fraction of the total and their removal is why the effective "
                "model count is not exactly the simulation count."
            )

        prediction_grid = prediction.reshape(rows, cols)
        variance_grid = variance.reshape(rows, cols)

        if hull is not None and hull["outside"] and mask_outside:
            prediction_grid = apply_hull_mask(
                prediction_grid, hull, DEFAULT_NODATA
            )
            variance_grid = apply_hull_mask(variance_grid, hull, DEFAULT_NODATA)

        feedback.pushInfo("Writing the predicted surface...")
        pred_path = self.parameterAsOutputLayer(parameters, self.PREDICTION, context)
        if not pred_path:
            raise QgsProcessingException(
                "No destination was given for the predicted surface."
            )
        try:
            write_float32_geotiff(pred_path, prediction_grid, extent,
                                  source.sourceCrs(), DEFAULT_NODATA,
                                  feedback=feedback)
        except (ValueError, RuntimeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

        var_path = self.parameterAsOutputLayer(parameters, self.VARIANCE, context)
        if var_path:
            feedback.pushInfo("Writing the prediction variance surface...")
            try:
                write_float32_geotiff(var_path, variance_grid, extent,
                                      source.sourceCrs(), DEFAULT_NODATA,
                                      feedback=feedback)
            except (ValueError, RuntimeError) as exc:
                raise QgsProcessingException(str(exc)) from exc
        else:
            feedback.pushInfo(
                "No destination was given for the prediction variance, so only "
                "the predicted surface was written. The variance is the part of "
                "this method that ordinary kriging cannot report."
            )

        subsets_id = self._write_subset_layer(
            parameters, context, source, coords_arr, detail, subset_info,
            feedback,
        )

        # The statistics are taken from the computed surfaces rather than the
        # masked ones, so the reported range describes what the method produced
        # instead of how much of it the hull mask removed.  The masked count is
        # reported separately and the hull section states the fraction.
        stats = {
            "prediction": surface_stats(prediction),
            "variance": surface_stats(variance),
            "cols": cols,
            "rows": rows,
            "cells": cells,
            "res_x": float(res_x),
            "res_y": float(res_y),
        }

        feedback.setProgress(95)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, model, alpha, drift, n_sims, n_bins,
            subset_size, overlap, seed, cv, cv_skip_note, stats,
            numeric_summary, crs_warning, detail, hull, mask_outside, var_path,
            max_points, work_note,
        )

        result = {
            self.PREDICTION: pred_path,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        if var_path:
            result[self.VARIANCE] = var_path
        if subsets_id:
            result[self.SUBSET_LAYER] = subsets_id
        feedback.setProgress(100)
        return result

    def _write_subset_layer(self, parameters, context, source, coords_arr,
                            detail, subset_info, feedback):
        """One point per subset carrying the model that was fitted there.

        Without this the automatic fitting is invisible: the report can say
        which families were chosen overall, but not where, and a subset that
        fell back to its initial fit looks exactly like one that did not.  The
        point is placed at the mean centre of the subset's observations, which
        is where that subset's variogram describes the field.
        """
        out_fields = subset_output_fields()
        sink, dest_id = self.parameterAsSink(
            parameters, self.SUBSET_LAYER, context, out_fields,
            QgsWkbTypes.Point, source.sourceCrs(),
        )
        if sink is None:
            return None

        feedback.pushInfo("Writing the per-subset fitted-model layer...")
        for index, (rows, summary) in enumerate(
                zip(subset_info["subsets"], detail["subsets"])):
            members = coords_arr[rows]
            feature = QgsFeature(out_fields)
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(
                float(members[:, 0].mean()), float(members[:, 1].mean())
            )))
            initial = summary["initial"]
            r2 = initial.get("r2")
            feature.setAttributes([
                int(index),
                str(summary["family"]),
                int(summary["n"]),
                float(initial["nugget"]),
                float(initial["psill"]),
                float(initial["range"]),
                float(r2) if r2 is not None and np.isfinite(r2) else None,
                float(summary["effective_models"]),
                float(summary["max_weight"]),
                int(summary["target_cells"]),
                1 if summary["fallback"] else 0,
            ])
            sink.addFeature(feature, QgsFeatureSink.FastInsert)
        del sink
        return dest_id

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(
        self, path, field_name, n, model, alpha, drift, n_sims, n_bins,
        subset_size, overlap, seed, cv, cv_skip_note, stats, numeric_summary,
        crs_warning, detail, hull, mask_outside, var_path, max_points, work_note,
    ):
        verdict, color, verdict_text, next_action = validation_verdict(
            cv, numeric_summary["std"] or 0.0, "Empirical Bayesian Kriging"
        )
        if cv is None:
            # The shared wording says an unvalidated method "carries no variance
            # model".  That is true of the interpolators it was written for and
            # false here, and a report that understates its own output is worse
            # than one that says nothing.
            verdict_text += (
                " This method does carry a variance model - the prediction "
                "variance surface is that model - but nothing in this run tested "
                "whether the variance matches the real error."
            )

        pred_stats = stats["prediction"]
        var_stats = stats["variance"]
        sd = numeric_summary["std"] or 0.0
        fit_flags, median_max_lag = _typical_fit(detail)
        median_bins = _median_of(detail["subsets"], "bins")

        # The calibrated-error reading is stated separately from the verdict,
        # because a run can predict well and still misstate its own error - and
        # that is exactly what the standardised errors are there to catch.
        mse = None if cv is None else cv["mse"]
        rmsse = None if cv is None else cv["rmsse"]
        if mse is None:
            variance_verdict = (
                "Nothing in this run tested the variance, so read it as a "
                "relative comparison between places rather than as an error bar."
            )
        elif abs(mse) <= 0.2 and 0.8 <= rmsse <= 1.2:
            variance_verdict = (
                f"Both standardised-error statistics are close to their ideal "
                f"values ({mse:+.3f} against 0, {rmsse:.3f} against 1), so this "
                "variance surface is neither over-stating nor under-stating the "
                "real prediction error on this field."
            )
        elif rmsse < 0.8:
            variance_verdict = (
                f"The root mean square standardised error is {rmsse:.3f} against "
                f"an ideal of 1, and the mean is {mse:+.3f} against an ideal of "
                "0. The variance surface is therefore smaller than the errors it "
                "is describing, so it will under-state the uncertainty of any "
                "decision made from it."
            )
        else:
            variance_verdict = (
                f"The root mean square standardised error is {rmsse:.3f} against "
                f"an ideal of 1, and the mean is {mse:+.3f} against an ideal of "
                "0. The variance surface is larger than the errors it is "
                "describing, so it will over-state the uncertainty of any "
                "decision made from it."
            )

        settings_pairs = [
            ("Requested family",
             EBK_MODEL_LABELS[EBK_MODEL_VALUES.index(model)]),
            ("Families actually fitted", ", ".join(detail["families"])),
            ("Stable shape exponent", f"{alpha:.6g}"),
            ("Trend removed before fitting", drift),
            ("Simulations per subset", f"{n_sims:,}"),
            ("Subset size requested",
             "one subset holding every observation" if subset_size == 0
             else f"{subset_size:,} observations"),
            ("Neighbour rings per subset", str(overlap)),
            ("Subsets used", f"{detail['n_subsets']:,}"),
            ("Subset sizes",
             f"{min(detail['subset_sizes'])} to {max(detail['subset_sizes'])} "
             "observations"),
            ("Distance bins requested, per fit", f"{n_bins:,}"),
            ("Random seed", f"{seed:,}"),
            ("Observation cap applied to this run", f"{max_points:,}"),
        ]

        ensemble_pairs = [
            ("Median effective model count", f"{detail['effective_models']:.2f}"),
            ("Smallest effective model count",
             f"{detail['min_effective_models']:.2f}"),
            ("Largest posterior weight on one model", f"{detail['max_weight']:.4f}"),
            ("Simulations discarded as non-convergent", f"{detail['n_failed']:,}"),
            ("Subsets that fell back to a single fit", f"{detail['fallbacks']:,}"),
            ("Largest Cholesky jitter used", f"{detail['jitter']:.3g}"),
            ("Median fitted range", fmt(fit_flags["range"])),
            ("Median fitted nugget", fmt(fit_flags["nugget"])),
            ("Median fitted partial sill", fmt(fit_flags["psill"])),
            ("Distance bins actually used",
             "not reported" if median_bins is None else f"{int(median_bins):,}"),
            ("Estimated computation", f"{detail['work_estimate']:,.0f} work units"),
        ]

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
            ("Variance minimum", fmt(var_stats["min"])),
            ("Variance maximum", fmt(var_stats["max"])),
            ("Variance mean", fmt(var_stats["mean"])),
        ]
        if hull is not None and mask_outside and hull["outside"]:
            surface_pairs.append(
                ("Cells written as no-data", f"{hull['outside']:,}")
            )

        if subset_size == 0:
            parallel_note = (
                "Every observation took part in every subset, so no locality was "
                "assumed and the fitted model describes the whole field. That is "
                "the cleaner estimator, and it is also the slower one: the cost "
                "grows with the observation count, which is the reason the method "
                "has subsets at all."
            )
        else:
            parallel_note = (
                f"The observations were split into {detail['n_subsets']:,} "
                f"subsets of about {subset_size:,}, each extended by {overlap} "
                "neighbouring ring(s). Every subset is fitted independently, so "
                "different families or very different ranges between subsets are "
                "a finding about the field rather than a failure of the fit - "
                "they mean the spatial structure is not the same everywhere. The "
                "per-subset layer carries each one's family and parameters if you "
                "want to map that."
            )

        method_note = (
            "Each subset's variogram is fitted, a new dataset is simulated from "
            "that fit at the same locations, the family is refitted to the "
            "simulated data, and the refit is weighted by the Gaussian "
            "likelihood of the real data under its parameters. Predictions from "
            "every weighted model are averaged. That loop is what makes the "
            "method empirical and Bayesian rather than a single fit: the models "
            "that could have produced these data are represented in proportion "
            "to how well they do so."
        )

        variance_note = (
            "The variance is the law of total variance over those weighted "
            "models, so it is never smaller than the average of their kriging "
            f"variances and it averages {var_stats['mean']:.6g} here. Measured "
            "against ordinary kriging fitted to the same data on the engine's "
            "synthetic test fields, the ratio is 1.11 to one overall and 1.28 to "
            "one within 25 map units of an observation: the inflation is largest "
            "where the observations are dense, because that is where the subsets "
            "disagree most about local structure. " + variance_verdict
        )
        if var_path:
            variance_note += (
                " The prediction and this variance were written as GeoTIFF "
                "Float32 rasters covering the extent of the input points; "
                f"non-finite cells carry the no-data value {DEFAULT_NODATA:g}."
            )
        else:
            variance_note += (
                " No destination was given for the variance surface, so only "
                "these summary statistics describe it."
            )

        cv_note = cv_protocol_note(cv)
        if cv_skip_note:
            cv_note += " " + cv_skip_note
        if cv is not None:
            cv_note += (
                " Leave-one-out was not attempted at all rather than attempted "
                "and abandoned: a held-out observation costs this method a "
                "complete refit - a simulated ensemble and a Cholesky "
                "factorisation per draw, in every subset - rather than the "
                "single kriging solve it costs ordinary kriging."
            )

        flags = quality_flags(fit_flags, None, median_max_lag)
        flag_html = ""
        if flags:
            items = "".join(f"<li>{html.escape(f)}</li>" for f in flags)
            flag_html = (
                '<section><h2>Automated Warnings</h2>'
                f'<ul class="flag-list">{items}</ul>'
                "<p>These are raised by fixed rules on the fitted numbers - "
                "here, on the median across the subsets, because this method "
                "fits one variogram per subset rather than one for the field. A "
                "warning therefore describes what a typical subset settled on "
                "and can be raised by a minority of them. Read them as prompts "
                "to check something, not as verdicts.</p></section>"
            )

        extrapolation_html = (
            "<section>"
            "<h2>Outside the Observed Area</h2>"
            f"<p>{html.escape(hull_note(hull, bool(mask_outside), DEFAULT_NODATA))}"
            "</p>"
            "<p>Empirical Bayesian Kriging has no hull constraint of its own: "
            "it returns a value and a variance at any target, including far "
            "outside the observations. Nothing in the method detects that it has "
            "left the sampled area, and the variance there is no substitute for "
            "one - it describes how well the fitted models agree with each "
            "other, which is a different question from whether the data support "
            "an extrapolation at all.</p>"
            "</section>"
        )

        work_html = ""
        if work_note:
            work_html = (
                '<section><h2>Computation Size</h2>'
                f"<p>{html.escape(work_note)}</p>"
                "<p>The cost of this method is the number of output cells "
                "multiplied by the number of simulations multiplied by the "
                "observations each simulation solves over. The estimate above is "
                "in the units the tool's own limits are measured in, and it is "
                "calibrated at the slow end of the range observed, so it "
                "over-states the wait rather than under-stating it.</p></section>"
            )

        guidance_html = analyst_guidance_html(
            "Empirical Bayesian Kriging Interpolation",
            "Fits a variogram per local subset, refits to simulated data, and "
            "predicts with the weighted ensemble - reporting a variance that "
            "includes the uncertainty in the variogram itself.",
            [
                "The cross-validation error is well below the field's own "
                "standard deviation.",
                "The standardised errors are close to their ideal values, so the "
                "variance surface describes the real prediction error.",
                "The effective model count is a substantial fraction of the "
                "requested simulation count, meaning the ensemble is genuinely "
                "averaging rather than collapsing onto one model.",
            ],
            [
                "Reading the variance surface as an absolute confidence "
                "interval. It is wider than ordinary kriging's on purpose, and "
                "it is conditional on the subset model being the right form.",
                "Interpolating far outside the extent of the observations. This "
                "method returns numbers there without complaint and without "
                "evidence.",
                "Choosing a single family for the whole area when the per-subset "
                "layer shows the fitted family changing from subset to subset. "
                "That pattern is information about the field, not noise.",
                "Comparing this cross-validation figure with a leave-one-out "
                "figure from ordinary kriging on the same points. The protocols "
                "differ and the numbers are not on the same scale.",
            ],
            [
                "Ordinary / Universal Kriging Interpolation on the same points, "
                "to see whether the extra model-uncertainty term changes any "
                "decision the map is being used for",
                "Empirical Semivariogram & Model Fitting to check the range and "
                "nugget the automatic fit settled on",
                "Inverse Distance Weighted Interpolation for a fast baseline "
                "with no variogram model at all",
            ],
            "Empirical Bayesian Kriging answers the same question as ordinary "
            "kriging and refuses to pretend the variogram is known. That makes it "
            "the right default when the field's structure is not already "
            "established and the reported uncertainty is going to be used for "
            "something. It is the wrong choice when the variogram is known "
            "independently and its value is in being enforced, because this "
            "method will quietly fit a different one where the data suggest it.",
        )

        caveat_text = [
            "The variance includes a term for the uncertainty in the variogram, "
            "so it is systematically wider than ordinary kriging's on the same "
            "data. Comparing the two surfaces without that difference in mind "
            "reads as a loss of precision when it is an honest gain in it.",
            "Five-fold cross-validation fits each model on four fifths of the "
            "data, so its error is larger than a leave-one-out figure would be. "
            "It is not comparable with leave-one-out numbers from other tools in "
            "this plugin, and the report states the protocol for that reason.",
            "The predictions are a mixture over models fitted to simulated data. "
            "They are reproducible from the seed, but a different seed moves them "
            "slightly - by well under the cross-validation error at the default "
            "simulation count.",
            "Where a subset could not refit enough models it falls back to a "
            "single initial fit, and its cells carry an ordinary kriging variance "
            "rather than a mixture variance. The count is reported above and the "
            "affected subsets are flagged in the per-subset layer.",
        ]

        summary_txt = (
            f"{n} observations were interpolated by Empirical Bayesian Kriging "
            f"over a {stats['cols']} x {stats['rows']} grid, "
            + ("fitting every subset to the whole field"
               if subset_size == 0 else
               f"fitting {detail['n_subsets']:,} local subsets")
            + f" with {n_sims:,} simulations each. Cross-validation error was "
            + ("<strong>not measured</strong>" if cv is None else
               f"<strong>{fmt(cv['rmse'], 6)}</strong>")
            + (f", which is {cv['nrmse']:.3f} of the field's standard deviation "
               f"of {sd:.6g}" if cv is not None and cv.get("nrmse") is not None
               else "")
            + "."
        )

        head = report_head(
            "Empirical Bayesian Kriging Interpolation",
            f"Field Interpolated: <strong>{html.escape(field_name)}</strong> | "
            f"Observations: <strong>{n}</strong> | Ensembles averaged: "
            f"<strong>{detail['n_subsets']:,} subsets x {n_sims:,} "
            "simulations</strong>",
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
            <tbody>{metric_rows(settings_pairs)}</tbody>
        </table>
        <p>{html.escape(parallel_note)}</p>
        <p>{html.escape(method_note)}</p>
    </section>

    <section>
        <h2>The Averaged Ensemble</h2>
        <table>
            <thead><tr><th>Ensemble Property</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(ensemble_pairs)}</tbody>
        </table>
        <p>The effective model count is the reciprocal of the sum of the squared
        posterior weights: it is the number of models the ensemble is really
        averaging, as opposed to the number it was given. It falls towards one
        when a single model dominates, at which point the extra simulations are
        buying nothing and the result is close to ordinary kriging under one
        fitted variogram.</p>
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
        <h2>Predicted Surface and Variance</h2>
        <table>
            <thead><tr><th>Surface Property</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(surface_pairs)}</tbody>
        </table>
        <p>{html.escape(variance_note)}</p>
    </section>

    {flag_html}

    {extrapolation_html}

    {work_html}

    {diagnostics_html(numeric_summary, None, crs_warning)}

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + body + report_foot(
                caveats_html("Empirical Bayesian Kriging", None,
                             numeric_summary, caveat_text),
                guidance_html,
            ))
