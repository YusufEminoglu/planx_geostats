# -*- coding: utf-8 -*-
"""Ordinary / Universal Kriging Interpolation Processing Algorithm."""
from __future__ import annotations

import html
import logging
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
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

from ..core.geostat_engines import (
    DRIFTS,
    KRIGING_SOFT_LIMIT,
    LOO_SOFT_LIMIT,
    MODELS,
    detrend,
    empirical_variogram,
    fit_variogram,
    fitted_curve,
    krige_chunked,
    loo_cross_validation,
    quality_flags,
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
    crs_unit_warning,
    diagnostics_html,
    numeric_quality_summary,
    push_diagnostics,
)
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import chart_css, semivariogram_svg
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")

MODEL_LABELS = ("Spherical", "Exponential", "Gaussian", "Linear", "Stable")

DRIFT_LABELS = (
    "None - ordinary kriging",
    "Linear trend - universal kriging",
    "Quadratic trend - universal kriging",
)

#: Observations needed before each drift order is worth attempting.  The drift
#: adds 3 columns (linear) or 6 (quadratic) to the kriging system, and a system
#: with fewer observations than unknowns is singular rather than merely
#: unstable, so these are floors rather than preferences.
_MIN_POINTS_FOR_DRIFT = {"none": 4, "linear": 10, "quadratic": 20}

#: Target chunk for the kriging solve.  The right-hand side is one double per
#: (observation + drift) pair per target, so this caps that allocation at about
#: 64 MB whatever the grid and point count.
_RHS_BYTE_BUDGET = 64 * 1024 * 1024


class KrigingAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    PARAM_SOURCE = "PARAM_SOURCE"
    PARAMS_LAYER = "PARAMS_LAYER"
    MODEL = "MODEL"
    ALPHA = "ALPHA"
    N_BINS = "N_BINS"
    DRIFT = "DRIFT"
    CELL_SIZE = "CELL_SIZE"
    MAX_POINTS = "MAX_POINTS"
    RUN_LOO = "RUN_LOO"
    PREDICTION = "PREDICTION"
    VARIANCE = "VARIANCE"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()

    def name(self) -> str:
        return "kriging_interpolation"

    def displayName(self) -> str:
        return "Ordinary / Universal Kriging Interpolation"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("kriging_interpolation")

    def createInstance(self):
        return KrigingAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Predicts a continuous surface from scattered point observations by "
            "kriging, using a fitted semivariogram model, and writes the kriging "
            "variance alongside it.\n\n"
            "Kriging is the interpolator that follows from a variogram rather "
            "than being chosen independently of it. Each prediction is a "
            "weighted average of the observations, where the weights come from "
            "solving a system built out of the fitted model: nearby points "
            "count for more, points that are clustered together are prevented "
            "from outvoting a lone one, and the solution is exact at the "
            "observed locations. Ordinary kriging assumes the field has a "
            "constant but unknown mean; universal kriging additionally fits a "
            "linear or quadratic trend, which is what to use when the values "
            "drift systematically across the area.\n\n"
            "The variance raster is the part worth understanding, because it is "
            "what separates this tool from an interpolator that only returns "
            "numbers. It is analytic, computed from the fitted model and the "
            "observation geometry, not estimated by resampling. Low variance "
            "means the observations around a cell constrain it tightly; high "
            "variance means they do not. It reports prediction uncertainty "
            "conditional on the model being correct - it cannot tell you that "
            "the model itself is wrong.\n\n"
            "Two properties of the variance surface look like faults and are "
            "not. Because the model is exact at a datum, the variance is zero "
            "there and rises away from observations. And for ordinary kriging "
            "the variance converges to the sill plus a non-negative "
            "mean-uncertainty term, so at the edge of the data the surface can "
            "legitimately exceed the fitted sill. Reaching the sill exactly is "
            "the behaviour of simple kriging, a different estimator. Read the "
            "variance surface as a relative statement about prediction error.\n\n"
            "Every run reports leave-one-out cross-validation - each observation "
            "predicted from all the others - so the model is judged on points it "
            "did not see. The mean error should be near zero, and the mean "
            "standardised error near zero with the root mean squared "
            "standardised error near one; values far from those mean the "
            "variogram is over- or under-stating the real prediction error.\n\n"
            "Cost grows sharply with the number of observations: the kriging "
            "system is solved once per run but its size is the observation "
            "count cubed. The default cap of 2000 observations is a deliberate "
            "ceiling; for reference, measured on the development machine, a "
            "200 x 200 surface took about 1.5 seconds with 200 observations, "
            "18 seconds with 800, and two minutes at 2000. Leave-one-out "
            "validation adds one solve per observation, so it is skipped above "
            "600 observations and the report says so.\n\n"
            "Output: a prediction raster and a kriging-variance raster, both "
            "GeoTIFF Float32 covering the extent of the input points, plus an "
            "HTML report carrying the model, the cross-validation statistics "
            "and the diagnostics."
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
                self.PARAM_SOURCE,
                "Where the semivariogram model comes from",
                options=[
                    "Fit a new model from this layer",
                    "Read the fitted-parameter layer from the semivariogram tool",
                ],
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PARAMS_LAYER,
                "Fitted semivariogram parameters (used only by the second option "
                "above)",
                [QgsProcessing.TypeVectorAnyGeometry],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODEL,
                "Theoretical semivariogram model (used when fitting)",
                options=list(MODEL_LABELS),
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
            QgsProcessingParameterNumber(
                self.N_BINS,
                "Number of distance bins for the internal fit",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=15,
                minValue=4,
                maxValue=200,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.DRIFT,
                "Trend removed before kriging",
                options=list(DRIFT_LABELS),
                defaultValue=0,
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
                "Maximum observations to krige with (cost is cubic in this "
                "number)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=KRIGING_SOFT_LIMIT,
                minValue=10,
                maxValue=200_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_LOO,
                "Run leave-one-out cross-validation (slow; adds one solve per "
                "observation)",
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
                "Kriging variance surface",
                optional=True,
                createByDefault=True,
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
                "HTML_REPORT_OUT", "Kriging interpolation report"
            )
        )

    # ── inputs ───────────────────────────────────────────────────────────────

    def _read_points(self, source, field_name, feedback):
        """Validated coordinates and values, plus the quality summary."""
        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        coords = []
        values = []
        value_by_id = {}
        missing_geometry = 0
        for f in source.getFeatures():
            if feedback.isCanceled():
                return None
            val = f.attribute(field_name)
            if val is None or val == NULL or str(val) == "NULL":
                continue
            try:
                z = float(val)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(z):
                continue
            centroid = geometry_centroid_point(f.geometry())
            if centroid is None:
                missing_geometry += 1
                continue
            coords.append([centroid.x(), centroid.y()])
            values.append(z)
            value_by_id[f.id()] = z

        summary = numeric_quality_summary(source.featureCount(), value_by_id, values)
        crs_warning = crs_unit_warning(source)
        if missing_geometry:
            feedback.pushWarning(
                f"{missing_geometry} feature(s) had no usable centroid and were "
                "excluded from the interpolation."
            )
        return coords, values, summary, crs_warning

    def _model_from_layer(self, parameters, context, feedback):
        """Read the five model parameters from the semivariogram tool's layer."""
        source = self.parameterAsSource(parameters, self.PARAMS_LAYER, context)
        if source is None:
            raise QgsProcessingException(
                "The semivariogram model is set to come from a fitted-parameter "
                "layer, but none was given. Either select one - the Empirical "
                "Semivariogram & Model Fitting tool writes it - or switch the "
                "model source back to fitting a new model."
            )

        names = {f.name() for f in source.fields()}
        required = {"model", "nugget", "psill", "range"}
        missing = sorted(required - names)
        if missing:
            raise QgsProcessingException(
                "The fitted-parameter layer is missing the field(s) "
                f"{', '.join(missing)}. Run Empirical Semivariogram & Model "
                "Fitting and use the layer that tool produces."
            )

        rows = list(source.getFeatures())
        if not rows:
            raise QgsProcessingException(
                "The fitted-parameter layer is empty, so there is no model to "
                "read. Re-run Empirical Semivariogram & Model Fitting."
            )
        if len(rows) > 1:
            feedback.pushWarning(
                f"The fitted-parameter layer holds {len(rows)} rows; the first "
                "is being used. Kriging needs one model, so keep only the row "
                "you mean to interpolate with."
            )

        feat = rows[0]
        model = str(feat["model"] or "").strip().lower()
        if model not in MODELS:
            raise QgsProcessingException(
                f"The fitted-parameter layer names the model '{model}', which "
                f"this tool does not implement. Expected one of: "
                f"{', '.join(MODELS)}."
            )

        def _number(name):
            raw = feat[name]
            if raw is None or raw == NULL or str(raw) == "NULL":
                raise QgsProcessingException(
                    f"The fitted-parameter layer has no value for '{name}'."
                )
            try:
                out = float(raw)
            except (TypeError, ValueError) as exc:
                raise QgsProcessingException(
                    f"The fitted-parameter layer's '{name}' is not a number: "
                    f"{raw!r}."
                ) from exc
            if not np.isfinite(out):
                raise QgsProcessingException(
                    f"The fitted-parameter layer's '{name}' is not finite."
                )
            return out

        nugget = _number("nugget")
        psill = _number("psill")
        rng = _number("range")
        alpha = 1.5
        if "alpha" in names:
            try:
                candidate = feat["alpha"]
                if candidate is not None and candidate != NULL and str(candidate) != "NULL":
                    alpha = float(candidate)
            except (TypeError, ValueError):
                alpha = 1.5
        if not np.isfinite(alpha):
            alpha = 1.5

        if nugget < 0:
            raise QgsProcessingException(
                f"The fitted-parameter layer's nugget is negative ({nugget})."
            )
        if psill <= 0 or rng <= 0:
            raise QgsProcessingException(
                "The fitted-parameter layer's partial sill and range must both "
                f"be positive; got partial sill {psill} and range {rng}."
            )
        return {
            "model": model,
            "nugget": nugget,
            "psill": psill,
            "sill": nugget + psill,
            "range": rng,
            "alpha": alpha,
        }, len(rows)

    def _model_from_fit(self, coords_arr, values_arr, parameters, context,
                        feedback, drift):
        """Fit a semivariogram to this layer, as the sibling tool would.

        With a trend requested, the variogram is fitted to the drift residuals
        rather than to the raw values -- see ``core.detrend`` for what happens
        to the fitted range when that step is skipped.
        """
        model = MODELS[self.parameterAsEnum(parameters, self.MODEL, context)]
        alpha = self.parameterAsDouble(parameters, self.ALPHA, context)
        n_bins = self.parameterAsInt(parameters, self.N_BINS, context)

        if drift == "none":
            fitted_values = values_arr
            detrend_note = None
        else:
            fitted_values, coefficients = detrend(coords_arr, values_arr, drift)
            detrend_note = _describe_drift(drift, coefficients)
            feedback.pushInfo(
                f"Removed a {drift} trend before fitting the variogram: "
                f"residual standard deviation {float(np.std(fitted_values)):.6g} "
                f"against {float(np.std(values_arr)):.6g} in the raw values."
            )

        lag, semivariance, npairs, _dist, _sv = empirical_variogram(
            coords_arr, fitted_values, n_bins=n_bins, max_lag=None
        )
        if len(lag) < 4:
            raise QgsProcessingException(
                f"Only {len(lag)} distance bin(s) contained point pairs, but "
                "fitting a semivariogram model needs at least 4. Increase the "
                "number of distance bins."
            )
        try:
            fitted = fit_variogram(lag, semivariance, npairs, model=model,
                                   alpha=alpha)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        feedback.pushInfo(
            f"Fitted {model}: nugget={fitted['nugget']:.6g}, "
            f"psill={fitted['psill']:.6g}, range={fitted['range']:.6g}, "
            f"R2={fitted['r2']:.4f}"
        )
        return {
            "model": model,
            "nugget": float(fitted["nugget"]),
            "psill": float(fitted["psill"]),
            "sill": float(fitted["sill"]),
            "range": float(fitted["range"]),
            "alpha": float(alpha),
            "r2": float(fitted["r2"]),
            "bins_used": int(fitted["bins_used"]),
            "detrend_note": detrend_note,
            "max_lag_used": float(lag[-1]) if len(lag) else None,
        }, (lag, semivariance, npairs)

    # ── run ──────────────────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
        drift = DRIFTS[self.parameterAsEnum(parameters, self.DRIFT, context)]
        run_loo = self.parameterAsBool(parameters, self.RUN_LOO, context)
        use_layer = self.parameterAsEnum(parameters, self.PARAM_SOURCE, context) == 1

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_kriging_report.html"
            )

        feedback.pushInfo("Reading point locations and target values...")
        read = self._read_points(source, field_name, feedback)
        if read is None:
            return {}
        coords, values, numeric_summary, crs_warning = read
        push_diagnostics(feedback, numeric_summary, None, crs_warning)

        n = len(values)
        if n < _MIN_POINTS_FOR_DRIFT["quadratic"] and drift == "quadratic":
            raise QgsProcessingException(
                f"Quadratic-trend kriging needs at least "
                f"{_MIN_POINTS_FOR_DRIFT['quadratic']} observations to fit a "
                f"6-term trend; got {n}. Use a linear trend or ordinary kriging."
            )
        if n < _MIN_POINTS_FOR_DRIFT[drift]:
            raise QgsProcessingException(
                f"{DRIFT_LABELS[DRIFTS.index(drift)]} needs at least "
                f"{_MIN_POINTS_FOR_DRIFT[drift]} observations; got {n}."
            )
        if numeric_summary["is_constant"]:
            raise QgsProcessingException(
                "The target field is constant, so every prediction would return "
                "that same value and the kriging variance would be undefined. "
                "There is nothing to interpolate."
            )
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. Kriging cost is cubic in the "
                "observation count, so the cap exists to keep the run bounded. "
                "Raise the cap deliberately if you accept the wait, or reduce "
                "the number of points first - the leave-one-out statistics this "
                "tool reports are the right guide to whether a sharper subset "
                "would change the surface."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        source_label = None
        plot = ""
        if use_layer:
            feedback.pushInfo("Reading the semivariogram model from the parameter layer...")
            model, row_count = self._model_from_layer(parameters, context, feedback)
            source_label = (
                "read from the fitted-parameter layer produced by Empirical "
                "Semivariogram & Model Fitting"
                + (f" ({row_count} rows present; the first was used)"
                   if row_count > 1 else "")
            )
            feedback.pushInfo(
                f"Using {model['model']}: nugget={model['nugget']:.6g}, "
                f"psill={model['psill']:.6g}, range={model['range']:.6g}"
            )
        else:
            feedback.pushInfo("Fitting a semivariogram model to this layer...")
            model, variogram_parts = self._model_from_fit(
                coords_arr, values_arr, parameters, context, feedback, drift
            )
            residual_fit = model["detrend_note"] is not None
            source_label = (
                f"fitted to this layer with the {model['model']} model "
                f"(weighted R2 = {model['r2']:.3f} over "
                f"{model['bins_used']} populated distance bins"
                + (", fitted to the drift residuals rather than the raw values"
                   if residual_fit else "")
                + ")"
            )
            lag, semivariance, npairs = variogram_parts
            h_fit, g_fit = fitted_curve(
                model["model"], model["nugget"], model["psill"], model["range"],
                float(lag[-1]) if len(lag) else 1.0, n=140, alpha=model["alpha"],
            )
            plot = semivariogram_svg(
                list(lag), list(semivariance),
                fitted=(list(h_fit), list(g_fit)),
                npairs=list(npairs),
                nugget=model["nugget"], sill=model["sill"],
                range_=model["range"], model_label=model["model"],
                y_label="Semivariance of drift residuals" if residual_fit
                else "Semivariance",
                title=("Residual semivariogram and the model used for kriging"
                       if residual_fit
                       else "Empirical semivariogram and the model used for "
                            "kriging"),
            )

        if feedback.isCanceled():
            return {}

        # ── leave-one-out validation, before the expensive part ──────────────
        loo = None
        loo_skipped = None
        if run_loo:
            if n > LOO_SOFT_LIMIT:
                loo_skipped = (
                    f"Leave-one-out validation was skipped: it costs one kriging "
                    f"solve per observation and this layer has {n}, above the "
                    f"limit of {LOO_SOFT_LIMIT}. Lower the observation count, or "
                    "turn the validation off to go straight to the surface."
                )
                feedback.pushWarning(loo_skipped)
            else:
                feedback.pushInfo(
                    f"Running leave-one-out cross-validation over {n} "
                    "observations (one kriging solve each)..."
                )
                try:
                    loo = loo_cross_validation(
                        coords_arr, values_arr, model["model"], model["nugget"],
                        model["psill"], model["range"], drift, model["alpha"],
                    )
                except ValueError as exc:
                    raise QgsProcessingException(str(exc)) from exc
                feedback.pushInfo(
                    f"Leave-one-out: ME={loo['me']:.6g}, RMSE={loo['rmse']:.6g}, "
                    f"MSE={loo['mse']:.6g}, RMSSE={loo['rmsse']:.6g}"
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
            f"Kriging a {cols} x {rows} grid ({cells:,} cells) at "
            f"{res_x:.6g} x {res_y:.6g} map units from {n} observations with a "
            f"{drift} drift."
        )
        if cells > 250_000 or n > 800:
            feedback.pushWarning(
                f"This run predicts {cells:,} cells from {n} observations, "
                "which can take several minutes. Progress is reported per chunk "
                "and the run can be cancelled."
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
            prediction, variance = krige_chunked(
                coords_arr, values_arr, targets, model["model"], model["nugget"],
                model["psill"], model["range"], drift, model["alpha"],
                max_rhs_bytes=_RHS_BYTE_BUDGET, progress=_progress,
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        except RuntimeError:
            return {}

        if feedback.isCanceled():
            return {}

        prediction_grid = prediction.reshape(rows, cols)
        variance_grid = variance.reshape(rows, cols)

        feedback.pushInfo("Writing the predicted surface...")
        pred_path = self.parameterAsOutputLayer(parameters, self.PREDICTION, context)
        if not pred_path:
            raise QgsProcessingException("No destination was given for the predicted surface.")
        try:
            write_float32_geotiff(pred_path, prediction_grid, extent,
                                  source.sourceCrs(), DEFAULT_NODATA, feedback=feedback)
        except (ValueError, RuntimeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

        var_path = self.parameterAsOutputLayer(parameters, self.VARIANCE, context)
        if var_path:
            feedback.pushInfo("Writing the kriging variance surface...")
            try:
                write_float32_geotiff(var_path, variance_grid, extent,
                                      source.sourceCrs(), DEFAULT_NODATA,
                                      feedback=feedback)
            except (ValueError, RuntimeError) as exc:
                raise QgsProcessingException(str(exc)) from exc
        else:
            feedback.pushInfo(
                "No destination was given for the kriging variance, so only the "
                "predicted surface was written."
            )

        stats = {
            "prediction": _surface_stats(prediction),
            "variance": _surface_stats(variance),
            "cols": cols,
            "rows": rows,
            "cells": cells,
            "res_x": float(res_x),
            "res_y": float(res_y),
        }

        feedback.setProgress(95)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, model, drift, source_label, loo,
            loo_skipped, stats, numeric_summary, crs_warning, plot,
            pred_path, var_path, max_points,
        )

        result = {
            self.PREDICTION: pred_path,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        if var_path:
            result[self.VARIANCE] = var_path
        feedback.setProgress(100)
        return result

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(
        self, path, field_name, n, model, drift, source_label, loo, loo_skipped,
        stats, numeric_summary, crs_warning, plot, pred_path, var_path,
        max_points,
    ):
        if loo is None:
            verdict, color = "Not validated", "#718096"
            desc = (
                loo_skipped or
                "Leave-one-out cross-validation was turned off for this run, so "
                "the surface below has not been tested against the observations "
                "it was built from. The kriging variance is still reported, but "
                "nothing here checks that it is calibrated."
            )
        else:
            rmse = loo["rmse"]
            sd = numeric_summary["std"] or 0.0
            good_mse = abs(loo["mse"]) <= 0.2
            good_rmsse = 0.8 <= loo["rmsse"] <= 1.2
            if good_mse and good_rmsse:
                verdict, color = "Validated", "#2f855a"
                desc = (
                    f"Leave-one-out prediction error is {rmse:.6g} in the "
                    f"field's own units, against a field standard deviation of "
                    f"{sd:.6g}. The mean standardised error ({loo['mse']:+.3f}) "
                    f"and the root mean squared standardised error "
                    f"({loo['rmsse']:.3f}) are both close to their ideal values "
                    "of 0 and 1, so the kriging variance is neither "
                    "over-stating nor under-stating the real prediction error."
                )
            elif good_mse or good_rmsse:
                verdict, color = "Partly validated", "#dd6b20"
                desc = (
                    f"Leave-one-out prediction error is {rmse:.6g}. The mean "
                    f"standardised error is {loo['mse']:+.3f} and the root mean "
                    f"squared standardised error is {loo['rmsse']:.3f}; ideal "
                    "values are 0 and 1. One of the two is off, so either the "
                    "predictions carry a systematic bias or the variance surface "
                    "misstates the error it is meant to describe. Read the "
                    "variance surface as relative rather than absolute."
                )
            else:
                verdict, color = "Poorly calibrated", "#c53030"
                desc = (
                    f"Leave-one-out prediction error is {rmse:.6g}, with a mean "
                    f"standardised error of {loo['mse']:+.3f} and a root mean "
                    f"squared standardised error of {loo['rmsse']:.3f} against "
                    "ideal values of 0 and 1. The variogram model does not "
                    "describe this field's structure well enough for the "
                    "variance surface to be taken at face value. Try the other "
                    "semivariogram models and a different drift order before "
                    "using these numbers."
                )

        flags = quality_flags(
            {"r2": model.get("r2"), "nugget": model["nugget"],
             "sill": model["sill"], "range": model["range"]},
            loo,
            max_lag=model.get("max_lag_used"),
        )
        flag_html = ""
        if flags:
            items = "".join(f"<li>{html.escape(f)}</li>" for f in flags)
            flag_html = (
                '<section><h2>Automated Warnings</h2>'
                f'<ul class="flag-list">{items}</ul>'
                "<p>These are raised by fixed rules on the fitted numbers, not "
                "by judgement about your specific data. Read them as prompts to "
                "check something, not as verdicts.</p></section>"
            )

        if loo is None:
            loo_rows = (
                '<tr><td class="metric-name">Leave-one-out validation</td>'
                '<td class="metric-val">not run</td></tr>'
            )
        else:
            sd = numeric_summary["std"] or 0.0
            nrmse = f"{loo['rmse'] / sd:.4f}" if sd > 0 else "n/a"
            loo_rows = "".join(
                "<tr>"
                f'<td class="metric-name">{label}</td>'
                f'<td class="metric-val">{value}</td>'
                "</tr>"
                for label, value in (
                    ("Observations validated", str(loo["n"])),
                    ("Mean error (ME)", f"{loo['me']:.6g}"),
                    ("Root mean squared error (RMSE)", f"{loo['rmse']:.6g}"),
                    ("Mean standardised error (MSE)", f"{loo['mse']:.6g}"),
                    ("RMS standardised error (RMSSE)", f"{loo['rmsse']:.6g}"),
                    ("RMSE / field standard deviation", nrmse),
                )
            )

        pred_stats = stats["prediction"]
        var_stats = stats["variance"]
        surface_rows = "".join(
            "<tr>"
            f'<td class="metric-name">{label}</td>'
            f'<td class="metric-val">{value}</td>'
            "</tr>"
            for label, value in (
                ("Output grid", f"{stats['cols']} x {stats['rows']} cells"),
                ("Cell size (map units)",
                 f"{stats['res_x']:.6g} x {stats['res_y']:.6g}"),
                ("Predicted minimum", f"{pred_stats['min']:.6g}"),
                ("Predicted maximum", f"{pred_stats['max']:.6g}"),
                ("Predicted mean", f"{pred_stats['mean']:.6g}"),
                ("Observed mean", f"{numeric_summary['mean']:.6g}"
                 if numeric_summary["mean"] is not None else "n/a"),
                ("Variance minimum", f"{var_stats['min']:.6g}"),
                ("Variance maximum", f"{var_stats['max']:.6g}"),
                ("Fitted sill for comparison", f"{model['sill']:.6g}"),
                ("Observation cap applied to this run", f"{max_points:,}"),
            )
        )

        drift_note = (
            f" The semivariogram was fitted to the residuals left after that "
            f"trend ({html.escape(model['detrend_note'])}); fitting it to the "
            "raw values instead would let the trend inflate the long-lag bins "
            "and push the fitted range towards the size of the study area."
            if model.get("detrend_note") else ""
        )

        variance_note = (
            "The variance is zero at every observation and grows with distance "
            "from them. Its maximum above can exceed the fitted sill of "
            f"{model['sill']:.6g}, which is correct rather than a fault: for "
            "ordinary kriging the variance converges to the sill plus a "
            "non-negative term for the uncertainty in the unknown mean, so it "
            "approaches the sill from above. Reaching the sill exactly is the "
            "behaviour of simple kriging. What the surface supports is a "
            "comparison between places - where the prediction is better and "
            "worse constrained - not a claim that the value at any one cell has "
            "a particular confidence interval."
        )
        if var_path:
            variance_note += (
                " The predicted surface and this variance were written as "
                "GeoTIFF Float32 rasters covering the extent of the input "
                "points; cells that could not be predicted carry the no-data "
                f"value {DEFAULT_NODATA:g}."
            )
        else:
            variance_note += (
                " No destination was given for the variance surface, so only "
                "these summary statistics describe it."
            )

        if loo is None:
            next_action = (
                "Re-run with leave-one-out validation switched on for a smaller "
                "subset of the observations. Without it there is nothing here "
                "that tests whether the variance surface describes the real "
                "prediction error."
            )
        elif abs(loo["mse"]) > 0.4 or loo["rmsse"] > 1.4 or loo["rmsse"] < 0.7:
            next_action = (
                "Treat the surface as indicative and revisit the variogram "
                "before relying on it. The standardised errors say the model "
                "misstates its own error; trying a different theoretical model, "
                "a different drift order, or a finer set of distance bins is "
                "the cheapest next step."
            )
        elif drift == "none":
            next_action = (
                "Check whether the residuals cluster geographically. A "
                "systematic pattern would mean universal kriging with a linear "
                "trend is a better description of this field than the constant "
                "mean ordinary kriging assumes."
            )
        else:
            next_action = (
                "Compare this surface against an ordinary kriging run on the "
                "same points. If the trended version does not reduce the "
                "leave-one-out error, the extra trend terms are not earning "
                "their complexity."
            )

        guidance_html = analyst_guidance_html(
            "Ordinary / Universal Kriging Interpolation",
            "Predicts a continuous surface from scattered observations using a "
            "fitted semivariogram model, and reports the analytic prediction "
            "variance alongside it.",
            [
                "The input layer is in a projected CRS, so distances entering "
                "the variogram are in metres or feet rather than degrees.",
                "The semivariogram model behind these predictions fits the "
                "empirical variogram well enough to describe the field.",
                "Leave-one-out statistics are close to their ideal values, so "
                "the variance surface is calibrated.",
            ],
            [
                "Reading the variance surface as an absolute confidence "
                "interval. It is conditional on the variogram model being "
                "correct.",
                "Interpolating far outside the extent of the observations. "
                "Kriging will still return numbers and a variance, but nothing "
                "in the data constrains them.",
                "Assuming a smooth surface is a well-supported one. A large "
                "nugget relative to the sill produces a smooth map from a field "
                "that has little structure at the sampled scale.",
            ],
            [
                "Empirical Semivariogram & Model Fitting to review the model "
                "these predictions rest on",
                "Local Moran's I to find where the observations themselves "
                "cluster, as a check on the predicted pattern",
                "Geographically Weighted Regression if the goal becomes "
                "explaining the field rather than mapping it",
            ],
            "Kriging answers where and how confidently, not why. Use the "
            "variance surface to decide where more sampling would pay off: the "
            "cells where it is highest are the ones the current observations "
            "constrain least.",
        )

        caveat_text = [
            "Kriging assumes the field is a realisation of a stationary random "
            "process described by the fitted semivariogram. The predictions are "
            "only as good as that assumption, and the reported variance cannot "
            "detect that it has failed.",
            "The variance surface is analytic - computed from the model and the "
            "observation geometry - and reports uncertainty only about the "
            "interpolation between existing observations, not about the data "
            "collection itself.",
            "Predictions outside the extent covered by the observations are "
            "extrapolation. The variance will warn you that they are weakly "
            "constrained, but it cannot say what the field does there.",
        ]

        model_rows = "".join(
            "<tr>"
            f'<td class="metric-name">{label}</td>'
            f'<td class="metric-val">{value}</td>'
            "</tr>"
            for label, value in (
                ("Theoretical model", html.escape(model["model"])),
                ("Nugget", f"{model['nugget']:.6g}"),
                ("Partial sill", f"{model['psill']:.6g}"),
                ("Sill", f"{model['sill']:.6g}"),
                ("Range (practical, map units)", f"{model['range']:.6g}"),
                ("Stable shape exponent (alpha)", f"{model['alpha']:.6g}"),
                ("Trend removed", html.escape(DRIFT_LABELS[DRIFTS.index(drift)])),
                ("Model source", html.escape(source_label)),
            )
        )

        r2_txt = ("n/a" if model.get("r2") is None
                  else f"{model['r2']:.6f}")
        loo_rmse_txt = "n/a" if loo is None else f"{loo['rmse']:.6g}"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Kriging Interpolation Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 860px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    .interpretation-box {{ background-color: #f8fafc; border-left: 5px solid {color}; padding: 20px; border-radius: 4px; margin-bottom: 30px; }}
    .status-title {{ font-size: 1.3rem; font-weight: 800; color: {color}; margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.05em; }}
    .status-desc {{ color: #4a5568; font-size: 0.95rem; margin: 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 25px; }}
    th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #edf2f7; font-size: 0.9rem; }}
    th {{ background-color: #ebf8ff; color: #2b6cb0; font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
    .metric-name {{ font-weight: 600; color: #2d3748; }}
    .metric-val {{ font-family: monospace; font-size: 1rem; font-weight: 600; }}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
    section {{ margin: 28px 0; }}
    h2 {{ color: #1a202c; font-size: 1.15rem; margin: 0 0 12px 0; }}
    .next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
    .variance-box {{ background: #fffaf0; border-left: 5px solid #dd6b20; padding: 16px 18px; border-radius: 4px; }}
    .flag-list {{ margin: 0 0 12px 0; padding-left: 22px; }}
    .flag-list li {{ margin-bottom: 8px; color: #4a5568; }}
    {analyst_guidance_css()}
    {chart_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Ordinary / Universal Kriging Interpolation</h1>
        <p class="subtitle">Field Interpolated: <strong>{html.escape(field_name)}</strong> | Observations: <strong>{n}</strong> | Model: <strong>{html.escape(model['model'])}</strong> | Trend: <strong>{html.escape(DRIFT_LABELS[DRIFTS.index(drift)])}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">Validation: {verdict}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>Kriging predicts each cell on the output grid as a weighted average of the surrounding observations. The weights are not a fixed function of distance: they come from solving a system built out of the semivariogram model, which is what lets kriging account for how the observations are arranged rather than only how far away they are. Where several observations sit close together, none of them is allowed to dominate simply by being one of a cluster. The prediction is exact at the observed locations. {html.escape(DRIFT_LABELS[DRIFTS.index(drift)])} was used, and the semivariogram model was {html.escape(source_label)}. The leave-one-out error over the observations was <strong>{loo_rmse_txt}</strong> in the field's own units.</p>
    </section>

    <section>
        <h2>Semivariogram Model Used</h2>
        <table>
            <thead><tr><th>Model Parameter</th><th>Value</th></tr></thead>
            <tbody>{model_rows}</tbody>
        </table>
        <p>The weighted goodness of fit of the model to its empirical variogram is <strong>{r2_txt}</strong>. It describes how well the curve follows the binned points, not how well kriging will predict, which is what the validation section below is for.{drift_note}</p>
    </section>
"""
        if plot:
            html_content += f"""
    <section>
        <h2>Fitted Semivariogram</h2>
        {plot}
    </section>
"""
        html_content += f"""
    <section>
        <h2>Leave-One-Out Cross-Validation</h2>
        <table>
            <thead><tr><th>Validation Statistic</th><th>Value</th></tr></thead>
            <tbody>{loo_rows}</tbody>
        </table>
        <p>Every observation is predicted from all the others, so these numbers describe how the model predicts points it has not seen rather than how well it fits the data it was fitted from. The mean error should be near zero: a large value means the surface is systematically high or low. The mean standardised error should be near zero and the root mean squared standardised error near one; together they say whether the kriging variance is honest about the error it produces. A root mean squared standardised error above one means real errors are larger than the variance claims.</p>
    </section>

    <section>
        <h2>Interpolated Surface</h2>
        <table>
            <thead><tr><th>Surface Property</th><th>Value</th></tr></thead>
            <tbody>{surface_rows}</tbody>
        </table>
        <div class="variance-box"><p style="margin:0;">{variance_note}</p></div>
    </section>

    {flag_html}

    {diagnostics_html(numeric_summary, None, crs_warning)}

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>

    {caveats_html("Kriging interpolation", None, numeric_summary, caveat_text)}

    {guidance_html}

    <footer>
        Generated by PlanX GeoStats Lab geostatistics engine.
    </footer>
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)


def _surface_stats(values):
    """Min/max/mean of a surface, ignoring non-finite cells."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def _describe_drift(drift, coefficients):
    """A readable statement of the trend that was removed, for the report."""
    c = [float(v) for v in coefficients]
    if drift == "linear":
        return (f"linear trend removed by least squares: "
                f"z = {c[0]:.6g} + {c[1]:.6g}*x + {c[2]:.6g}*y")
    if drift == "quadratic":
        return (f"quadratic trend removed by least squares: "
                f"z = {c[0]:.6g} + {c[1]:.6g}*x + {c[2]:.6g}*y + "
                f"{c[3]:.6g}*x^2 + {c[4]:.6g}*y^2 + {c[5]:.6g}*x*y")
    return "no trend removed"
