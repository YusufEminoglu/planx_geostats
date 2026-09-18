# -*- coding: utf-8 -*-
"""Indicator Kriging Processing Algorithm.

Threshold-exceedance probability surfaces, not a value surface.  The question
this tool answers is "how likely is this cell to be above this level", asked
once per level, and it answers it without assuming anything about the
distribution of the field: the observation is replaced by a 0/1 indicator and
kriged, so the only model involved is the one the analyst can see in the
report.

The tool writes three surfaces, and the order of the bands in the first one is
part of its contract rather than an implementation detail - PyQGIS cannot name
a band, so if the report did not say which band is which, nothing would.
"""
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
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterString,
    QgsRectangle,
)

from ..core.indicator_engine import (
    DEFAULT_THRESHOLD_COUNT,
    IK_DEFAULT_FOLDS,
    IK_LOO_LIMIT,
    IK_MIN_POINTS,
    IK_MODELS,
    IK_SOFT_LIMIT,
    MIN_THRESHOLD_SEPARATION,
    calibration_table,
    check_workload,
    class_means,
    correct_order_relations,
    declustered_frequencies,
    exceedance_probabilities,
    etype_from_probabilities,
    exact_indicator_curve,
    fit_indicator_model,
    indicator_cross_validation,
    indicator_is_constant,
    indicator_krige,
    indicator_vector,
    max_lag_for,
    median_parameter,
    parse_thresholds,
    threshold_separation,
)
from ..core.raster_io import (
    DEFAULT_NODATA,
    auto_cell_size,
    grid_centres,
    grid_shape,
    write_float32_geotiff,
    write_float32_multiband_geotiff,
)
from ..core.analysis_diagnostics import (
    caveats_html,
    diagnostics_html,
    push_diagnostics,
)
from ..core.interpolation_report import (
    apply_hull_mask,
    fmt,
    hull_note,
    hull_summary,
    metric_rows,
    report_foot,
    report_head,
    surface_stats,
)
from ..core.reporting import analyst_guidance_html

from ._icons import algorithm_icon
from ._input import read_point_values, require_non_constant


logger = logging.getLogger("PlanX GeoStats Lab")

#: The tool's model enum.  "Automatic" first, for the same reason EBK puts it
#: first: the reader who does not already know the field's variogram is the one
#: this tool is written for.  Labels and values are both generated from the
#: engine's tuple so the two cannot drift apart.
IK_MODEL_VALUES = ("auto",) + tuple(IK_MODELS)
IK_MODEL_LABELS = ("Automatic - whichever family fits best",) + tuple(
    name.capitalize() for name in IK_MODELS
)

#: A cross-validation fold needs enough observations that each fold's training
#: half can still fit a variogram; the engine refuses below twice its minimum,
#: so the tool checks the same floor rather than discovering it from a raise.
CV_MIN_OBSERVATIONS = IK_MIN_POINTS * 2

#: The Brier skill score bands the verdict is read from.  A skill score of zero
#: means the surface predicts no better than always quoting the threshold's own
#: observed frequency and ignoring location entirely, which is the bar being
#: applied; one means a perfect forecast.
SKILL_GOOD = 0.30
SKILL_MODEST = 0.10

#: Equal-count bins for the calibration table.  Five is enough to show a curve
#: that is systematically over-confident in its middle without slicing the
#: held-out scores so thin that each bin is noise.
CALIBRATION_BINS = 5


def probability_verdict(cv):
    """Turn a Brier cross-validation into a verdict, a colour and a sentence.

    The metric is deliberately not the RMSE the value interpolators report.  A
    Brier score is a squared error in *probability* units, and the scale it is
    judged on is the base rate rather than the field's standard deviation: a
    surface that cannot beat the threshold's own observed frequency has learned
    nothing spatial, however small its error looks.

    Returns ``(verdict, color, description, next_action)``.
    """
    if cv is None:
        return (
            "Not validated", "#718096",
            "Cross-validation was turned off for this run, so nothing here "
            "measures whether the probabilities below are calibrated by "
            "location. The surfaces are still produced, and the E-type estimate "
            "still averages them, but a probability surface nobody has scored "
            "is a drawing.",
            "Re-run with cross-validation switched on. The Brier score and its "
            "skill score are the only measures in this report that test whether "
            "the probabilities mean anything.",
        )

    skill = cv["brier_skill_score"]
    mean_brier = cv["mean_brier"]
    reference = cv["reference_brier"]
    exact_note = ""
    if cv["n_thresholds_exact"]:
        exact_note = (
            f" {cv['n_thresholds_exact']} threshold(s) lay outside the observed "
            "range, where the answer is exact rather than estimated, and are "
            "reported separately rather than scored."
        )
    if not np.isfinite(skill):
        return (
            "Not comparable", "#718096",
            "Cross-validation ran, but every threshold was answered exactly "
            "rather than estimated, so there is no forecast error to judge."
            + exact_note,
            "Every threshold you asked for lies outside the observed range of "
            "the field. Choose thresholds inside it to get probabilities that "
            "depend on location.",
        )

    base = (
        f"The Brier score is {mean_brier:.4f} against {reference:.4f} for "
        f"always quoting each threshold's own observed frequency, a skill score "
        f"of {skill:+.4f}. "
    )
    if skill >= SKILL_GOOD:
        return (
            "Probabilities are sharp and calibrated", "#2f855a",
            base + "The surfaces beat the base rate by a wide margin, which "
            "means the exceedance probabilities vary with location in a way the "
            "observations support rather than restating the overall frequency.",
            "Compare against a threshold you can check by hand - pick the "
            "median and confirm the 0.5 band follows the mapped pattern - "
            "before using the probabilities as a decision input.",
        )
    if skill >= SKILL_MODEST:
        return (
            "Better than the base rate", "#dd6b20",
            base + "The surfaces carry real spatial information, but not a "
            "great deal of it: choosing thresholds the data separate better, or "
            "adding observations near the levels you care about, is where the "
            "gain is.",
            "Read the per-threshold scores below rather than the average. One "
            "threshold usually carries most of the skill, and the others are "
            "close to the base rate.",
        )
    if skill > 0.0:
        return (
            "Barely better than the base rate", "#dd6b20",
            base + "There is almost no spatial information here. The surfaces "
            "will still look like surfaces, which is why this number is "
            "reported rather than left out.",
            "Check whether the field has any spatial structure at all - the "
            "semivariogram tool answers that directly - before spending "
            "thresholds' worth of computation on it.",
        )
    return (
        "No better than the base rate", "#c53030",
        base + "A skill score at or below zero means the probabilities do no "
        "better than quoting the overall frequency at every cell, and may do "
        "worse. The maps are still drawn; they are not evidence."
        + exact_note,
        "Do not use these probabilities as a decision input. Check the field "
        "for spatial structure, and check that the thresholds you chose sit "
        "where the observations can actually separate them.",
    )


def weak_fit_indices(fits):
    """Indices of the thresholds whose fitted variogram explains nothing.

    A threshold whose fitted variogram scores an R2 at or below zero against its
    own mean has no spatial structure at that level for the kriging to use, so
    its surface is close to that threshold's overall frequency everywhere.

    This is a module-level function rather than an expression written twice
    because it is needed in two places - the run log, which warns while the fit
    is fresh, and the report, which is written afterwards from the fits
    themselves.  As a local of the run it reached the report as a free variable
    and raised NameError there, so the report could only ever be produced by a
    run whose editor had not noticed.  The flake8 gate caught it as F821 on
    ``weak_fits``; nothing else would have, because the line is only reached
    once a report is actually written.
    """
    return [
        index for index, fit in enumerate(fits)
        if fit is not None and fit.get("r2") is not None
        and float(fit["r2"]) <= 0.0
    ]


def probability_cv_rows(cv) -> str:
    """The per-threshold validation table body for an indicator run."""
    if cv is None:
        return (
            '<tr><td class="metric-name">Cross-validation</td>'
            '<td class="metric-val">not run</td></tr>'
        )
    rows = []
    for index, row in enumerate(cv["thresholds"]):
        if row["exact"]:
            rows.append((
                f"Threshold {index + 1} ({fmt(row['threshold'])})",
                f"exact - {row['reason']}",
            ))
        elif row["n"] == 0:
            rows.append((
                f"Threshold {index + 1} ({fmt(row['threshold'])})",
                "not scored - no held-out observation could be predicted",
            ))
        else:
            rows.append((
                f"Threshold {index + 1} ({fmt(row['threshold'])})",
                f"Brier {row['brier']:.4f} | mean predicted P(Z &lt;= z) "
                f"{row['mean_probability']:.4f} against the observed "
                f"at-or-below frequency {row['observed_freq']:.4f}",
            ))
    return "".join(
        "<tr>"
        f'<td class="metric-name">{html.escape(label)}</td>'
        f'<td class="metric-val">{html.escape(value)}</td>'
        "</tr>"
        for label, value in rows
    )


class IndicatorKrigingAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    THRESHOLDS = "THRESHOLDS"
    THRESHOLD_COUNT = "THRESHOLD_COUNT"
    MODEL = "MODEL"
    ALPHA = "ALPHA"
    N_BINS = "N_BINS"
    MAX_LAG = "MAX_LAG"
    CELL_SIZE = "CELL_SIZE"
    MAX_POINTS = "MAX_POINTS"
    RUN_CV = "RUN_CV"
    CV_FOLDS = "CV_FOLDS"
    MASK_OUTSIDE = "MASK_OUTSIDE"
    EXCEEDANCE = "EXCEEDANCE"
    ETYPE = "ETYPE"
    ETYPE_VARIANCE = "ETYPE_VARIANCE"
    HTML_REPORT = "HTML_REPORT"

    def name(self) -> str:
        return "indicator_kriging"

    def displayName(self) -> str:
        return "Indicator Kriging (Threshold Exceedance Probability)"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("indicator_kriging")

    def createInstance(self):
        return IndicatorKrigingAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Answers 'how likely is this location to be above this level', "
            "once per level you name, without assuming anything about how the "
            "field's values are distributed.\n\n"
            "The usual way to get a probability from a kriged surface is to "
            "assume the field is normal around the prediction, which is an "
            "assumption about the data that the data does not have to satisfy. "
            "Indicator kriging does not make it. Each threshold becomes a 0/1 "
            "variable - is this observation at or below the level - and that "
            "variable is kriged on its own. The prediction is then the "
            "probability of being at or below the level, by construction rather "
            "than by assumption, and the price is one variogram per threshold.\n\n"
            "You get three surfaces. The exceedance raster has one band per "
            "threshold, band 1 being the smallest threshold, so band k holds the "
            "chance of exceeding your k-th threshold. The bands therefore fall "
            "as the band number rises: the highest band is the rarest event. "
            "The band order cannot be written into the file - PyQGIS has no way "
            "to name a band - so the report and this help text are the contract, "
            "and the report repeats the order in full.\n\n"
            "The second surface is the E-type estimate: the expected value of "
            "the field, rebuilt from the probabilities. It is the surface to "
            "compare with an ordinary kriging prediction, and it will differ, "
            "because it is optimal for a different loss. Read the report's note "
            "before treating it as an interpolator. It is not exact at the "
            "observations, even though the probabilities underneath it are: at "
            "a datum the fitted curve is a perfect step, so the expected value "
            "lands on the middle of the class the observation falls in rather "
            "than on the observation itself. That gap shrinks as you add "
            "thresholds - but only for observations in the middle of the range. "
            "The observations above your highest threshold and below your "
            "lowest sit in open-ended classes that no extra threshold can "
            "narrow, and the report says so with the measured numbers.\n\n"
            "Leave the threshold list empty and the tool places five quantiles "
            "of the data for you. Name your own and they are used as given: a "
            "regulatory limit is someone else's number and replacing it with a "
            "quantile of your sample would answer a different question.\n\n"
            "The thresholds are then repaired if their curves cross - a "
            "fitted probability is not automatically monotone in the threshold, "
            "and a set of numbers that falls somewhere is not a distribution. "
            "The repair is reported by size, and a large repair is a finding: it "
            "means your thresholds are closer together than the sampling can "
            "resolve.\n\n"
            "Validation is a Brier score, not an RMSE, with a skill score "
            "against the base rate and a calibration table. A threshold outside "
            "the observed range is answered exactly rather than estimated, and "
            "is reported separately so its trivially perfect score cannot lift "
            "the average."
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
                "Target numeric field",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.THRESHOLDS,
                "Thresholds to exceed, comma or semicolon separated (empty = "
                "quantiles of the data)",
                defaultValue="",
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.THRESHOLD_COUNT,
                "Thresholds to place when none are given (quantiles of the "
                "observed values, which carry no external meaning)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=DEFAULT_THRESHOLD_COUNT,
                minValue=1,
                maxValue=50,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODEL,
                "Semivariogram family for the indicators (Automatic fits every "
                "threshold to all of them and keeps the best)",
                options=list(IK_MODEL_LABELS),
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
                "Number of distance bins for each threshold's variogram",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=15,
                minValue=4,
                maxValue=200,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_LAG,
                "Maximum lag for the variogram in map units (0 = automatic)",
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
                "Maximum observations this run will accept (cost is their "
                "count multiplied by the cell count and the threshold count)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=IK_SOFT_LIMIT,
                minValue=IK_MIN_POINTS,
                maxValue=200_000,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RUN_CV,
                "Score the probabilities by cross-validation (Brier score and "
                "skill score against the base rate)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.CV_FOLDS,
                "Cross-validation folds (used only when leave-one-out is not "
                "affordable at this observation count)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=IK_DEFAULT_FOLDS,
                minValue=2,
                maxValue=20,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.MASK_OUTSIDE,
                "Write cells outside the observations' convex hull as no-data "
                "(probabilities are still produced there, but nothing "
                "constrains them)",
                defaultValue=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.EXCEEDANCE,
                "Threshold exceedance probabilities (one band per threshold, "
                "band 1 = smallest threshold)",
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.ETYPE,
                "E-type estimate of the field (expected value rebuilt from the "
                "probabilities; not an exact interpolator)",
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.ETYPE_VARIANCE,
                "E-type variance (optional; not the kriging variance)",
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
                "HTML_REPORT_OUT", "Indicator kriging report"
            )
        )

    # ── run ──────────────────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        threshold_text = self.parameterAsString(parameters, self.THRESHOLDS,
                                                context)
        threshold_count = self.parameterAsInt(
            parameters, self.THRESHOLD_COUNT, context
        )
        model = IK_MODEL_VALUES[self.parameterAsEnum(parameters, self.MODEL,
                                                     context)]
        alpha = self.parameterAsDouble(parameters, self.ALPHA, context)
        n_bins = self.parameterAsInt(parameters, self.N_BINS, context)
        max_lag = self.parameterAsDouble(parameters, self.MAX_LAG, context)
        cell_size = self.parameterAsDouble(parameters, self.CELL_SIZE, context)
        max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
        run_cv = self.parameterAsBool(parameters, self.RUN_CV, context)
        folds = self.parameterAsInt(parameters, self.CV_FOLDS, context)
        mask_outside = self.parameterAsBool(
            parameters, self.MASK_OUTSIDE, context
        )

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT,
                                               context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_indicator_kriging_report.html"
            )

        feedback.pushInfo("Reading point locations and target values...")
        read = read_point_values(source, field_name, feedback)
        if read is None:
            return {}
        coords, values, numeric_summary, crs_warning = read
        push_diagnostics(feedback, numeric_summary, None, crs_warning)

        n = len(values)
        if n < IK_MIN_POINTS:
            raise QgsProcessingException(
                f"Indicator kriging needs at least {IK_MIN_POINTS} "
                f"observations with a finite value and a usable geometry, so "
                f"that each threshold leaves both classes represented; got "
                f"{n}."
            )
        require_non_constant(numeric_summary)
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. The cost is the observation "
                "count multiplied by the cell count and by the number of "
                "thresholds, so it is the cheapest of the three to reduce if "
                "the thresholds are the point of the run. Raise the cap "
                "deliberately if you accept the wait."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        try:
            thresholds, threshold_source = parse_thresholds(
                threshold_text, values_arr, threshold_count
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        smallest_gap, relative_gap = threshold_separation(thresholds, values_arr)
        if np.isfinite(relative_gap) and relative_gap < MIN_THRESHOLD_SEPARATION:
            feedback.pushWarning(
                f"The two closest thresholds are {smallest_gap:.6g} apart, "
                f"which is {relative_gap:.2e} of the field's range. Their "
                "indicators differ on very few observations, so the two "
                "probability bands are the same curve fitted to noise and the "
                "order-relation repair will pool them."
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
                f"Cell size defaulted to {cell_size:.6g} map units, giving "
                "about 200 cells on the longer side."
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

        resolved_max_lag = max_lag_for(coords_arr, max_lag)
        n_thresholds = len(thresholds)
        try:
            work_note = check_workload(cells, n, n_thresholds)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        feedback.pushInfo(
            f"Evaluating a {cols} x {rows} grid ({cells:,} cells) from {n} "
            f"observations at {n_thresholds} threshold(s), each solved as its "
            "own kriging system."
        )
        if work_note:
            feedback.pushWarning("This run is large: " + work_note)
        feedback.pushInfo(
            "Thresholds: " + ", ".join(f"{t:.6g}" for t in thresholds)
            + (f" ({threshold_source})" if threshold_source else "")
        )

        # ── cross-validation, before the expensive part ──────────────────────
        cv = None
        cv_skip_note = None
        if run_cv:
            if n < CV_MIN_OBSERVATIONS:
                cv_skip_note = (
                    f"Cross-validation was skipped: a {folds}-fold split of "
                    f"{n} observations would leave fewer than {IK_MIN_POINTS} "
                    "to fit each threshold's variogram on. Add observations, or "
                    "lower the fold count. Nothing in this report then tests "
                    "whether the probabilities are calibrated."
                )
                feedback.pushWarning(cv_skip_note)
            else:
                feedback.pushInfo(
                    f"Scoring the probabilities by cross-validation over {n} "
                    f"observations and {n_thresholds} threshold(s), refitting "
                    "every threshold's variogram in every fold..."
                )

                def _cv_progress(done, total):
                    feedback.setProgress(int(60.0 * done / max(1, total)))
                    return not feedback.isCanceled()

                try:
                    cv = indicator_cross_validation(
                        coords_arr, values_arr, thresholds, model=model,
                        alpha=alpha, n_bins=n_bins, max_lag=resolved_max_lag,
                        folds=folds, progress=_cv_progress,
                    )
                except ValueError as exc:
                    raise QgsProcessingException(str(exc)) from exc
                except RuntimeError:
                    return {}
                if np.isfinite(cv["brier_skill_score"]):
                    feedback.pushInfo(
                        f"Cross-validation ({cv['protocol']}): mean Brier "
                        f"{cv['mean_brier']:.4f} against a base rate of "
                        f"{cv['reference_brier']:.4f}, skill "
                        f"{cv['brier_skill_score']:+.4f}"
                    )
                else:
                    feedback.pushInfo(
                        f"Cross-validation ({cv['protocol']}): no threshold "
                        "could be scored, so there is no skill score to report."
                    )
                if cv["n_thresholds_exact"]:
                    feedback.pushInfo(
                        f"{cv['n_thresholds_exact']} threshold(s) lay outside "
                        "the observed range and were answered exactly rather "
                        "than scored. The report says which."
                    )
                if cv["n_dropped"]:
                    feedback.pushWarning(
                        f"{cv['n_dropped']} of the held-out scores this run "
                        "attempted could not be produced, because a fold's "
                        "training half carried only one indicator class. They "
                        "are excluded from the score rather than counted as "
                        "correct."
                    )
                if feedback.isCanceled():
                    return {}

        # ── one kriging system per threshold ─────────────────────────────────
        total_chunks = max(1, n_thresholds)

        def _progress(index):
            """Map a threshold's own chunk reporting into the run's progress."""
            def _inner(done, total):
                fraction = (index + done / max(1, total)) / total_chunks
                feedback.setProgress(int(60.0 + 35.0 * fraction))
                return not feedback.isCanceled()
            return _inner

        estimates = []
        variances = []
        fits = []
        exact_flags = []
        overshoot_total = 0
        for index, threshold in enumerate(thresholds):
            if feedback.isCanceled():
                return {}
            ind = indicator_vector(values_arr, threshold)
            if indicator_is_constant(ind):
                estimate, variance = exact_indicator_curve(ind, len(targets))
                fits.append(None)
                exact_flags.append(True)
                feedback.pushInfo(
                    f"Threshold {index + 1} ({threshold:.6g}) lies outside the "
                    "observed range: every observation is on the same side of "
                    "it, so its probability is exact and no variogram was "
                    "fitted."
                )
            else:
                try:
                    fit, _lag, _sem, _pairs = fit_indicator_model(
                        coords_arr, ind, model=model, alpha=alpha,
                        n_bins=n_bins, max_lag=resolved_max_lag,
                    )
                except ValueError as exc:
                    raise QgsProcessingException(
                        f"Threshold {index + 1} ({threshold:.6g}) could not be "
                        f"fitted: {exc}"
                    ) from exc
                estimate, variance, overshoot = indicator_krige(
                    coords_arr, ind, targets, fit, progress=_progress(index),
                )
                overshoot_total += overshoot
                fits.append(fit)
                exact_flags.append(False)
            estimates.append(estimate)
            variances.append(variance)

        # A threshold whose fitted variogram explains no more than the
        # indicator's own mean has no spatial structure at that level for the
        # kriging to use, and its surface is close to the threshold's overall
        # frequency everywhere.  That is worth saying out loud: the run's
        # verdict is driven by the pooled skill score, so one flat threshold
        # lowers the average without announcing itself, and R2 at or below zero
        # is the line where the fit has earned nothing.
        weak_fits = weak_fit_indices(fits)
        if weak_fits:
            listed = ", ".join(
                f"{index + 1} ({thresholds[index]:.6g})"
                for index in weak_fits
            )
            feedback.pushWarning(
                f"The indicator variogram of threshold(s) {listed} fits no "
                "better than the indicator's own mean: R2 is at or below zero, "
                "so there is no spatial structure at that level, over this lag "
                "range, for the kriging to use. That threshold's surface is "
                "close to its overall frequency everywhere rather than varying "
                "with location, and its calibration band in the report will "
                "show near-identical predicted probabilities."
            )

        stacked = np.vstack(estimates)
        corrected, repair = correct_order_relations(stacked)

        if repair["cells"]:
            feedback.pushInfo(
                f"The order-relation repair changed {repair['cells']:,} of "
                f"{n_thresholds * cells:,} probability cell(s), pooling up to "
                f"{repair['max_block']} adjacent thresholds into one, and moved "
                f"no probability by more than "
                f"{repair['max_adjustment']:.6g}."
            )
        if overshoot_total:
            feedback.pushWarning(
                f"{overshoot_total:,} raw probability estimate(s) fell outside "
                "[0, 1] and were clipped into range before the repair. A "
                "probability outside its own range is ordinary from a kriging "
                "system and is reported rather than hidden; a large count means "
                "the fitted variogram is a poor description of the indicator."
            )

        exceedance = exceedance_probabilities(corrected)
        means, edges, empty_classes = class_means(values_arr, thresholds)
        etype, etype_variance, _class_probabilities = etype_from_probabilities(
            corrected, means
        )

        if empty_classes:
            feedback.pushInfo(
                f"{len(empty_classes)} of the {len(thresholds) + 1} value "
                "classes hold no observation; their means are interval "
                "midpoints, which the report states."
            )

        exceedance_grid = exceedance.reshape(n_thresholds, rows, cols)
        etype_grid = etype.reshape(rows, cols)
        variance_grid = etype_variance.reshape(rows, cols)

        masked_cells = 0
        if hull is not None and hull["outside"] and mask_outside:
            masked_cells = int(hull["outside"])
            exceedance_grid = np.stack([
                apply_hull_mask(exceedance_grid[k], hull, DEFAULT_NODATA)
                for k in range(n_thresholds)
            ])
            etype_grid = apply_hull_mask(etype_grid, hull, DEFAULT_NODATA)
            variance_grid = apply_hull_mask(variance_grid, hull, DEFAULT_NODATA)

        # The fraction of the *mapped area* at or below a threshold and the
        # fraction of the *observations* that are, are two different numbers as
        # soon as the sampling is clustered, and ordinary kriging's far field
        # answers the first - it converges to the declustered mean, which is why
        # it beats a distance-weighted average.  A surface whose band averages
        # 0.86 beside a sample frequency of 0.75 therefore reads as a broken
        # output to anyone who compares the two, and nothing in the report said
        # which of them the surface was answering.  Both are computed here so
        # the two can be shown together, with the declustered frequency between
        # them as the value the surface should be landing on.
        band_area_means = 1.0 - np.asarray([
            surface_stats(exceedance_grid[k], DEFAULT_NODATA)["mean"]
            for k in range(n_thresholds)
        ], dtype=float)
        band_sample_freqs = np.asarray(
            [(values_arr <= t).mean() for t in thresholds], dtype=float
        )
        try:
            decluster = declustered_frequencies(
                coords_arr, values_arr, thresholds
            )
        except ValueError:
            decluster = None

        # ── write the surfaces ───────────────────────────────────────────────
        band_labels = [
            f"P({field_name} > {threshold:.6g})" for threshold in thresholds
        ]
        exceedance_path = self.parameterAsOutputLayer(
            parameters, self.EXCEEDANCE, context
        )
        if not exceedance_path:
            raise QgsProcessingException(
                "No destination was given for the exceedance probabilities, "
                "which are this tool's whole output."
            )
        feedback.pushInfo(
            f"Writing {n_thresholds} exceedance band(s): band 1 is the smallest "
            f"threshold ({thresholds[0]:.6g}) and band {n_thresholds} the "
            f"largest ({thresholds[-1]:.6g})."
        )
        try:
            write_float32_multiband_geotiff(
                exceedance_path, list(exceedance_grid), extent,
                source.sourceCrs(), DEFAULT_NODATA, band_labels=band_labels,
                feedback=feedback,
            )
        except (ValueError, RuntimeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

        etype_path = self.parameterAsOutputLayer(parameters, self.ETYPE, context)
        if not etype_path:
            raise QgsProcessingException(
                "No destination was given for the E-type estimate."
            )
        feedback.pushInfo("Writing the E-type estimate...")
        try:
            write_float32_geotiff(etype_path, etype_grid, extent,
                                  source.sourceCrs(), DEFAULT_NODATA,
                                  feedback=feedback)
        except (ValueError, RuntimeError) as exc:
            raise QgsProcessingException(str(exc)) from exc

        variance_path = self.parameterAsOutputLayer(
            parameters, self.ETYPE_VARIANCE, context
        )
        if variance_path:
            feedback.pushInfo("Writing the E-type variance...")
            try:
                write_float32_geotiff(variance_path, variance_grid, extent,
                                      source.sourceCrs(), DEFAULT_NODATA,
                                      feedback=feedback)
            except (ValueError, RuntimeError) as exc:
                raise QgsProcessingException(str(exc)) from exc

        # Summarised from the grids that were written, not from the unmasked
        # arrays behind them.  The two differ whenever the hull mask is on, and
        # the difference is not cosmetic: the report states a no-data cell count
        # beside these figures, so statistics taken over the cells that count as
        # no-data describe a different surface from the one in the file - and a
        # reader comparing them against the band table, which is taken over the
        # written cells, has no way to see why the two disagree.
        stats = {
            "exceedance": surface_stats(exceedance_grid, DEFAULT_NODATA),
            "etype": surface_stats(etype_grid, DEFAULT_NODATA),
            "etype_variance": surface_stats(variance_grid, DEFAULT_NODATA),
            "cols": cols,
            "rows": rows,
            "cells": cells,
            "res_x": float(res_x),
            "res_y": float(res_y),
            "masked_cells": masked_cells,
        }

        feedback.setProgress(96)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, thresholds, threshold_source, model,
            alpha, n_bins, resolved_max_lag, max_lag, cv, cv_skip_note, stats,
            numeric_summary, crs_warning, fits, exact_flags, repair,
            overshoot_total, means, empty_classes, hull, mask_outside,
            band_labels, variance_path, max_points, work_note,
            band_area_means, band_sample_freqs, decluster,
        )

        result = {
            self.EXCEEDANCE: exceedance_path,
            self.ETYPE: etype_path,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        if variance_path:
            result[self.ETYPE_VARIANCE] = variance_path
        feedback.setProgress(100)
        return result

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(
        self, path, field_name, n, thresholds, threshold_source, model, alpha,
        n_bins, resolved_max_lag, requested_max_lag, cv, cv_skip_note, stats,
        numeric_summary, crs_warning, fits, exact_flags, repair,
        overshoot_total, means, empty_classes, hull, mask_outside, band_labels,
        variance_path, max_points, work_note,
        band_area_means, band_sample_freqs, decluster,
        calibration_bins=CALIBRATION_BINS,
    ):
        verdict, color, verdict_text, next_action = probability_verdict(cv)

        exceed_stats = stats["exceedance"]
        etype_stats = stats["etype"]
        variance_stats = stats["etype_variance"]
        fitted = [fit for fit in fits if fit is not None]

        settings_pairs = [
            ("Thresholds used", ", ".join(f"{t:.6g}" for t in thresholds)),
            ("Where the thresholds came from",
             "named by the analyst" if threshold_source == "given"
             else "quantiles of this sample, so they carry no external meaning"),
            ("Requested variogram family",
             IK_MODEL_LABELS[IK_MODEL_VALUES.index(model)]),
            ("Families actually fitted",
             ", ".join(sorted({fit["model"] for fit in fitted}))
             if fitted else "none - every threshold was answered exactly"),
            ("Stable shape exponent", f"{alpha:.6g}"),
            ("Distance bins per variogram", f"{n_bins:,}"),
            ("Maximum lag",
             f"{resolved_max_lag:.6g} map units"
             + (" (automatic)" if requested_max_lag <= 0 else " (requested)")),
            ("Observation cap applied to this run", f"{max_points:,}"),
        ]

        threshold_pairs = []
        for index, threshold in enumerate(thresholds):
            label = f"Threshold {index + 1} ({fmt(threshold)})"
            if exact_flags[index]:
                threshold_pairs.append((
                    label,
                    "outside the observed range - answered exactly, no "
                    "variogram fitted",
                ))
                continue
            fit = fits[index]
            row = cv["thresholds"][index] if cv is not None else None
            mean_predicted = row["mean_probability"] if row is not None else None
            share = (f" | mean predicted P(Z <= z) {mean_predicted:.4f}"
                     if mean_predicted is not None
                     and np.isfinite(mean_predicted) else "")
            r2 = fit.get("r2")
            threshold_pairs.append((
                label,
                f"{fit['model']}, nugget {fmt(fit['nugget'])}, partial sill "
                f"{fmt(fit['psill'])}, range {fmt(fit['range'])}"
                + (f", R2 {r2:.4f}" if r2 is not None else "")
                + share,
            ))

        n_estimated = len(thresholds) - int(np.count_nonzero(exact_flags))
        parameter_pairs = [
            ("Median fitted nugget", fmt(median_parameter(fitted, "nugget"))),
            ("Median fitted partial sill",
             fmt(median_parameter(fitted, "psill"))),
            ("Median fitted range", fmt(median_parameter(fitted, "range"))),
            ("Thresholds estimated", f"{n_estimated} of {len(thresholds)}"),
            ("Thresholds answered exactly",
             str(int(np.count_nonzero(exact_flags)))),
        ]

        surface_pairs = [
            ("Output grid", f"{stats['cols']} x {stats['rows']} cells"),
            ("Cell size (map units)",
             f"{stats['res_x']:.6g} x {stats['res_y']:.6g}"),
            ("Exceedance bands written", f"{len(thresholds):,}"),
            ("Lowest exceedance probability", fmt(exceed_stats["min"])),
            ("Highest exceedance probability", fmt(exceed_stats["max"])),
            ("Mean exceedance probability", fmt(exceed_stats["mean"])),
            ("E-type minimum", fmt(etype_stats["min"])),
            ("E-type maximum", fmt(etype_stats["max"])),
            ("E-type mean", fmt(etype_stats["mean"])),
            ("Observed minimum", fmt(numeric_summary["minimum"])),
            ("Observed maximum", fmt(numeric_summary["maximum"])),
            ("Observed mean", fmt(numeric_summary["mean"])),
            ("E-type variance minimum", fmt(variance_stats["min"])),
            ("E-type variance maximum", fmt(variance_stats["max"])),
        ]
        if stats["masked_cells"]:
            surface_pairs.append(
                ("Cells written as no-data", f"{stats['masked_cells']:,}")
            )

        repair_pairs = [
            ("Cells the repair changed",
             f"{repair['cells']:,} of {len(thresholds) * stats['cells']:,}"),
            ("Largest number of thresholds pooled into one probability",
             str(repair["max_block"])),
            ("Largest change the repair made to a probability",
             f"{repair['max_adjustment']:.6g}"),
            # Counted where the estimates are produced, not read back out of
            # the repair: ``indicator_krige`` clips before returning, so the
            # array handed to the repair can never hold a value outside [0, 1]
            # and a count taken there is zero whatever the data did.  It was
            # reported that way and read as "nothing was clipped" beside a
            # settings row saying 2 207 of them had been.
            ("Raw estimates outside [0, 1], clipped into range",
             f"{overshoot_total:,} of {len(thresholds) * stats['cells']:,}"),
        ]

        class_pairs = []
        for index, mean in enumerate(means):
            if index == 0:
                span = f"at or below {fmt(thresholds[0])}"
            elif index == len(thresholds):
                span = f"above {fmt(thresholds[-1])}"
            else:
                span = f"{fmt(thresholds[index - 1])} to {fmt(thresholds[index])}"
            marker = (
                " (no observations; interval midpoint)"
                if index in empty_classes else ""
            )
            class_pairs.append((f"Class {index + 1}: {span}",
                                f"{fmt(mean)}{marker}"))

        band_pairs = [
            (f"Band {index + 1}", label)
            for index, label in enumerate(band_labels)
        ]

        n_exact = int(np.count_nonzero(exact_flags))
        exact_note = ""
        if n_exact:
            exact_note = (
                f" {n_exact} of the {len(thresholds)} thresholds lie outside "
                "the observed range. Their indicator never varies, so their "
                "probability is 0 or 1 everywhere by arithmetic rather than by "
                "estimation, and cross-validation reports them separately and "
                "leaves them out of the average. A band that is uniformly 1 or "
                "uniformly 0 is the correct answer, not a failed fit."
            )

        method_note = (
            "Each threshold turns the observations into a 0/1 variable - was "
            "this observation at or below the level - and that variable is "
            "kriged on its own by ordinary kriging. The result is the "
            "probability of being at or below the threshold, by construction "
            "rather than by assuming a distribution for the field. Ordinary "
            "kriging is used rather than simple kriging because the mean of the "
            "indicator is exactly what is being estimated, so it cannot be "
            "supplied in advance."
        )

        repair_note = (
            "Probabilities fitted threshold by threshold are not guaranteed to "
            "rise with the threshold, and a set of numbers that falls somewhere "
            "is not a distribution. Wherever two adjacent bands were out of "
            "order they were replaced by their average, which is the standard "
            "repair, and it conserves each cell's total probability. It is "
            "reported by size rather than applied silently because a large "
            "repair is a finding: it means the thresholds are closer together "
            "than the sampling can tell apart, so those bands are really one "
            "probability rather than several."
        )

        etype_note = (
            "The E-type estimate is the expected value of the field rebuilt "
            "from the probabilities: the sum over classes of each class's mean "
            "value times the probability of falling in it. Two things about it "
            "will look wrong and are not. It is not the kriging variance - it "
            "is the variance of the value implied by the probability curve, and "
            "the two answer different questions. And it is not an exact "
            "interpolator, although the probabilities underneath it are: at an "
            "observation the fitted curve is a perfect step, so the expected "
            "value lands on the mean of the class that observation falls in "
            "rather than on the observation. That gap is bounded by the width "
            "of the class and shrinks as thresholds are packed more densely - "
            "but only for observations in the middle of the range. The "
            "observations above the highest threshold and below the lowest sit "
            "in open-ended classes that no extra threshold can narrow, so the "
            "tails of the E-type surface stay the least trustworthy part of it "
            "however many thresholds you spend."
        )
        # The two means sit two rows apart in the surface table, and on a skewed
        # field they can differ by a factor of two, which reads as a broken
        # output unless the report says what it is.  The larger part of the gap
        # is not an error at all and is not the class representatives: E-type
        # averages the class probabilities over the map, and those probabilities
        # describe the *area*, while the observed mean is a mean over the
        # *sample*.  Clustered sampling - which is the usual case, since
        # observations follow roads and survey lines - makes those two different
        # numbers, and the band table above shows by how much.  The
        # class-representative effect is real but it is the smaller term, and
        # only the first term scales with the gap between the sample frequency
        # and the declustered one.
        observed_mean = numeric_summary["mean"]
        etype_mean = etype_stats["mean"]
        if (np.isfinite(observed_mean) and np.isfinite(etype_mean)
                and observed_mean != 0.0
                and abs(etype_mean - observed_mean) > 0.1 * abs(observed_mean)):
            etype_mean_note = (
                "The E-type mean here is "
                f"{fmt(etype_mean)} against an observed mean of "
                f"{fmt(observed_mean)}, a gap of "
                f"{abs(etype_mean - observed_mean) / abs(observed_mean):.0%}. "
                "Most of that is not a miscalibrated probability. The "
                "probabilities underneath are averaged over the mapped area, "
                "and the observed mean is an average over the sample; when the "
                "sample is clustered the two answer different questions, and "
                "the band table above shows which. The second and smaller part "
                "is the class-representative effect: at an observation the "
                "fitted curve is a perfect step, so the E-type returns that "
                "observation's class mean rather than the observation, and a "
                "field whose classes are unequal in width is pulled toward the "
                "class means. Read the probabilities directly for decisions at "
                "a threshold; read the E-type as a smoothed reconstruction, and "
                "add thresholds before reading its level as a value."
            )
        else:
            etype_mean_note = (
                "The E-type mean is close to the observed mean, which is what "
                "a well-calibrated probability curve should give; neither the "
                "declustering gap nor the class-representative effect that pulls "
                "this surface low on a skewed field is doing much here."
            )

        # The level of the bands, stated against both reference frequencies.
        # The table exists because the surface legitimately answers the area
        # rather than the sample, so a reader who compares a band average of
        # 0.86 against a sample frequency of 0.75 has no way to tell a correct
        # declustering from a broken probability unless both numbers are on the
        # page with the declustered one between them.
        band_level_pairs = []
        for index, threshold in enumerate(thresholds):
            area = band_area_means[index]
            sample = band_sample_freqs[index]
            row = [f"Threshold {index + 1} ({fmt(threshold)})",
                   f"mapped area {fmt(area, 4)} | sample {fmt(sample, 4)}"]
            if decluster is not None:
                row[1] += (f" | declustered "
                           f"{fmt(float(decluster['median'][index]), 4)}")
            band_level_pairs.append(tuple(row))

        if decluster is None:
            band_level_note = (
                "The mapped fraction is the area average of the cumulative "
                "probability surface; the sample fraction is the proportion of "
                "the observations at or below the threshold. Those differ "
                "whenever the sampling is clustered relative to the mapped "
                "extent, and ordinary kriging's far field tracks the first, "
                "because it converges on the declustered mean of the data "
                "rather than the plain one. The declustered frequency could not "
                "be computed for this run, so the two numbers here are reported "
                "without it."
            )
        else:
            worst = int(np.argmax(np.abs(
                np.asarray(band_area_means, dtype=float)
                - np.asarray(decluster["median"], dtype=float)
            )))
            spread = float(np.max(decluster["high"] - decluster["low"]))
            band_level_note = (
                "The mapped fraction is the area average of the cumulative "
                "probability surface; the sample fraction is the proportion of "
                "observations at or below the threshold; the declustered "
                "frequency reweights the observations so that one alone in a "
                "cell counts as much as a crowded cell, and is the value the "
                "surface should be landing on. Ordinary kriging's far field "
                "converges on the declustered mean, so a mapped fraction near "
                "the declustered one and away from the sample one is the method "
                "working, not a defect. The declustering sweep is "
                f"{decluster['low'].min():.4f} to {decluster['high'].max():.4f} "
                f"across cell sizes of "
                f"{decluster['cell_sizes'].min():.6g} to "
                f"{decluster['cell_sizes'].max():.6g} map units, a spread of "
                f"{spread:.4f}, and the busiest cell holds "
                f"{int(decluster['counts'].max())} observation(s), which is how "
                "clustered this sample is. The threshold whose mapped fraction "
                f"is furthest from its declustered frequency is "
                f"threshold {worst + 1} "
                f"({fmt(abs(band_area_means[worst] - float(decluster['median'][worst])), 4)} "
                "apart), and a gap there is worth reading against this spread "
                "before it is read as an error."
            )

        cv_note = ""
        if cv is None:
            cv_note = cv_skip_note or (
                "Cross-validation was not run, so nothing here tests the "
                "probabilities."
            )
        else:
            if cv["protocol_id"] == "loo":
                protocol_txt = (
                    "Every observation was held out in turn and its probability "
                    "predicted from the rest, refitting that threshold's "
                    "variogram each time. That refit is the cost and the point: "
                    "borrowing the variogram from the full dataset would score "
                    "the fit as well as the prediction. Leave-one-out is the "
                    "strongest test available and the only one whose result does "
                    "not depend on a random split."
                )
            else:
                protocol_txt = (
                    f"The observations were split into {cv['folds']} groups and "
                    "each group predicted from the rest, refitting every "
                    "threshold's variogram inside each fold. One refit per fold "
                    "per threshold is what makes leave-one-out unaffordable "
                    f"above {IK_LOO_LIMIT} observations - the cap this run was "
                    f"held to, not this run's {n}. Each model therefore sees "
                    "only a "
                    "fraction of the data, so this error is systematically "
                    "larger than a leave-one-out figure would be and is not "
                    "comparable with one. The split is fixed by a seed, so the "
                    f"{len(cv['held_out_predicted']):,} held-out scores behind "
                    "this figure are reproducible."
                )
            cv_note = (
                protocol_txt + " The metric is the Brier score, the mean "
                "squared error in probability units, because a probability is "
                "what is being predicted and an error in the field's own units "
                "would not be comparable between thresholds. The skill score "
                "measures it against the base rate - always quoting each "
                "threshold's observed frequency and ignoring location - so a "
                "skill of zero means the surface has learned nothing spatial, "
                "however small its Brier score looks." + exact_note
            )
        if cv_skip_note and cv is not None:
            cv_note += " " + cv_skip_note
        if cv is not None and cv["n_dropped"]:
            cv_note += (
                f" {cv['n_dropped']} of the held-out scores this run attempted "
                "could not be produced, because a fold's training half carried "
                "only one indicator class. They are excluded rather than "
                "counted as correct."
            )

        calibration_html = self._calibration_html(cv, calibration_bins)

        extrapolation_html = (
            "<section>"
            "<h2>Outside the Observed Area</h2>"
            f"<p>{html.escape(hull_note(hull, bool(mask_outside), DEFAULT_NODATA))}"
            "</p>"
            "<p>Indicator kriging has no hull constraint of its own: it returns "
            "a probability at any target, including far outside the "
            "observations, and a probability surface reads as confident "
            "everywhere. The probabilities beyond the sampled area are an "
            "extrapolation of the indicator's variogram and nothing in the "
            "method flags them.</p>"
            "</section>"
        )

        work_html = ""
        if work_note:
            work_html = (
                '<section><h2>Computation Size</h2>'
                f"<p>{html.escape(work_note)}</p>"
                "<p>Every threshold is a separate kriging system: its own "
                "variogram, its own matrix factorisation, its own solve over the "
                "whole grid. That is why the threshold count multiplies the cost "
                "rather than adding to it, and why dropping a threshold you do "
                "not need is the cheapest way to make a run affordable.</p>"
                "</section>"
            )

        if threshold_source == "quantiles":
            # The band description has to come from the threshold count this
            # run actually used: the default is five, so the wording was
            # written as "fifths" and then read a three-threshold run as
            # fifths too, which is a wrong statement about the output sitting
            # in the report beside the right number of bands.
            n_bands = len(thresholds) + 1
            band_word = {
                2: "halves", 3: "thirds", 4: "quarters", 5: "fifths",
                6: "sixths", 7: "sevenths", 8: "eighths", 9: "ninths",
                10: "tenths", 11: "elevenths", 12: "twelfths",
            }.get(n_bands)
            bands = (
                f"the sample's own {band_word}"
                if band_word else
                f"{n_bands} equal-frequency bands of this sample"
            )
            threshold_note = (
                "The thresholds were placed at quantiles of this sample, so the "
                f"bands are {bands} and carry no meaning outside "
                "it. That is the honest default when no level has been given to "
                "you, but a threshold that matters is usually someone's number - "
                "a limit, a target, a class boundary - and if you have one, name "
                "it instead: the tool will use it as given."
            )
        else:
            threshold_note = (
                "The thresholds were used exactly as given, in ascending order. "
                "That matters because a regulatory limit or a planning target is "
                "a specific number chosen for reasons outside this analysis, and "
                "replacing it with a quantile of the local data would answer a "
                "different question from the one that was asked."
            )

        guidance_html = analyst_guidance_html(
            "Indicator Kriging",
            "Produces a probability surface per threshold without assuming a "
            "distribution for the field, plus the expected value rebuilt from "
            "those probabilities.",
            [
                "The Brier skill score is positive and the calibration table "
                "shows predicted probabilities tracking observed frequencies.",
                "The threshold list is the one the decision uses, not a set of "
                "quantiles standing in for it.",
                "The order-relation repair is small enough that the thresholds "
                "are resolved by this sampling rather than pooled by it.",
            ],
            [
                "Reading the E-type surface as an exact interpolator. It is "
                "exact at no observation, and its error is largest in the tails "
                "of the distribution whatever you do.",
                "Treating the E-type variance as a confidence interval. It is "
                "the spread of the value implied by the probability curve, not "
                "an error bar for the prediction.",
                "Adding thresholds indiscriminately. Each one costs a variogram "
                "and a full kriging system, and thresholds closer together than "
                "the sampling can resolve are pooled by the repair rather than "
                "resolved by it.",
                "Comparing the Brier score with an RMSE from another tool. They "
                "are errors in different units and are not on the same scale; "
                "the skill score against the base rate is the comparable "
                "number.",
            ],
            [
                "Empirical Semivariogram & Model Fitting to see whether the "
                "field has the spatial structure these probabilities assume",
                "Ordinary / Universal Kriging Interpolation to compare the "
                "E-type surface with a value prediction on the same points",
                "The interpolation model comparison tool to see how the "
                "methods score side by side on this field",
            ],
            "Indicator kriging is the right tool when the question is a "
            "probability and the level is fixed - the chance of exceeding a "
            "flood level, a pollution limit, a density target - and the wrong "
            "one when the question is what the value is. If you want a value, "
            "the E-type surface here is a way of comparing the probabilities, "
            "not a substitute for kriging the values directly.",
        )

        caveat_text = [
            "The probabilities are only as good as the thresholds. Thresholds "
            "closer together than the observations can resolve produce "
            "indicators that differ on a handful of points, and the repair "
            "pools them rather than separating them. The repair table above "
            "says how much of that happened.",
            "The E-type estimate is not an exact interpolator even though the "
            "probability curve it is built from is. At an observation it "
            "returns the mean of that observation's class, and the gap is "
            "only narrowed by adding thresholds - which does nothing at all in "
            "the open-ended tail classes.",
            "Indicator kriging makes no distributional assumption, but it does "
            "assume each indicator has its own stationary variogram and that "
            "the single fitted model describes it everywhere in the extent. "
            "The per-threshold parameters above are the evidence for that, and "
            "a poor fit is visible in the R2 column. A good R2 is not the "
            "converse: it says the fitted curve tracks the binned cloud, not "
            "that the indicator has a sill inside this lag range. Where it does "
            "not - the semivariance still climbing at the largest lag, which is "
            "what a spatial trend at that threshold looks like - the fitted "
            "partial sill can exceed the indicator's own variance and the "
            "kriging has no sill to shrink to, so beyond the observations it "
            "extrapolates the trend instead of returning to the overall "
            "frequency. The clipped-estimate count in the surface table is "
            "where that shows up.",
            "The Brier score is a squared error in probability units and is not "
            "comparable with the RMSE figures the value interpolators in this "
            "plugin report. The skill score against the base rate is the number "
            "to compare across runs.",
            "Cross-validation refits every threshold's variogram in every fold, "
            "so at high observation counts the tool switches from leave-one-out "
            "to grouped folds and says so. A grouped figure is systematically "
            "worse than a leave-one-out one and the two are not interchangeable.",
        ]
        weak_fits = weak_fit_indices(fits)
        if weak_fits:
            listed = ", ".join(
                f"{index + 1} ({thresholds[index]:.6g})"
                for index in weak_fits
            )
            caveat_text.append(
                f"The indicator variogram of threshold(s) {listed} has an R2 at "
                "or below zero, so at that level the observations carry no "
                "spatial structure this lag range can resolve. The kriging "
                "still returns a surface there, and it is very close to that "
                "threshold's overall frequency at every location: read it as a "
                "constant with noise rather than as a map. The skill score "
                "above is the average over all thresholds and does not single "
                "this one out."
            )

        if cv is None:
            summary_txt = (
                f"{n} observations were turned into {len(thresholds)} "
                f"threshold exceedance surfaces over a {stats['cols']} x "
                f"{stats['rows']} grid. Cross-validation was "
                "<strong>not run</strong>, so nothing here says whether those "
                "probabilities are calibrated."
            )
        elif np.isfinite(cv["brier_skill_score"]):
            summary_txt = (
                f"{n} observations were turned into {len(thresholds)} "
                f"threshold exceedance surfaces over a {stats['cols']} x "
                f"{stats['rows']} grid. Cross-validation says the probabilities "
                f"beat the base rate with a Brier skill score of "
                f"<strong>{cv['brier_skill_score']:+.4f}</strong>, on a Brier "
                f"score of {fmt(cv['mean_brier'])} against "
                f"{fmt(cv['reference_brier'])} for ignoring location."
            )
        else:
            summary_txt = (
                f"{n} observations were turned into {len(thresholds)} "
                "threshold exceedance surfaces, all of them outside the "
                "observed range and therefore answered exactly rather than "
                "estimated. There is no forecast error to score."
            )

        head = report_head(
            "Indicator Kriging (Threshold Exceedance Probability)",
            f"Field Interpolated: <strong>{html.escape(field_name)}</strong> | "
            f"Observations: <strong>{n}</strong> | Thresholds: "
            f"<strong>{len(thresholds)}</strong>",
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
        <p>{html.escape(method_note)}</p>
        <p>{html.escape(threshold_note)}</p>
    </section>

    <section>
        <h2>Thresholds and Their Fitted Variograms</h2>
        <table>
            <thead><tr><th>Threshold</th><th>Fitted Model</th></tr></thead>
            <tbody>{metric_rows(threshold_pairs)}</tbody>
        </table>
        <table>
            <thead><tr><th>Across the Estimated Thresholds</th><th>Value</th>
            </tr></thead>
            <tbody>{metric_rows(parameter_pairs)}</tbody>
        </table>
        <p>One variogram per threshold is the price of making no distributional
        assumption.{html.escape(exact_note)}</p>
    </section>

    <section>
        <h2>Order Relations</h2>
        <table>
            <thead><tr><th>Repair Statistic</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(repair_pairs)}</tbody>
        </table>
        <p>{html.escape(repair_note)}</p>
    </section>

    <section>
        <h2>Cross-Validation of the Probabilities</h2>
        <table>
            <thead><tr><th>Threshold</th><th>Result</th></tr></thead>
            <tbody>{probability_cv_rows(cv)}</tbody>
        </table>
        <p>{html.escape(cv_note)}</p>
        {calibration_html}
    </section>

    <section>
        <h2>Level of the Probability Bands</h2>
        <table>
            <thead><tr><th>Threshold</th><th>Cumulative Probability at or
            Below It</th></tr></thead>
            <tbody>{metric_rows(band_level_pairs)}</tbody>
        </table>
        <p>{html.escape(band_level_note)}</p>
    </section>

    <section>
        <h2>Value Classes and the E-Type Estimate</h2>
        <table>
            <thead><tr><th>Class</th><th>Mean Value Used</th></tr></thead>
            <tbody>{metric_rows(class_pairs)}</tbody>
        </table>
        <p>{html.escape(etype_note)}</p>
        <p>{html.escape(etype_mean_note)}</p>
    </section>

    <section>
        <h2>Output Surfaces</h2>
        <table>
            <thead><tr><th>Surface Property</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(surface_pairs)}</tbody>
        </table>
        <p>The exceedance raster holds one band per threshold and nothing in
        the file names them, so this is the order. Band k holds the
        probability of exceeding the k-th threshold, with the thresholds
        ascending, which means the probabilities fall as the band number
        rises:</p>
        <table>
            <thead><tr><th>Band</th><th>Meaning</th></tr></thead>
            <tbody>{metric_rows(band_pairs)}</tbody>
        </table>
        <p>{html.escape(self._surface_note(variance_path))}</p>
    </section>

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
                caveats_html("Indicator Kriging", None, numeric_summary,
                             caveat_text),
                guidance_html,
            ))

    def _calibration_html(self, cv, bins):
        """Calibration: predicted probability binned against what happened.

        A Brier score says how wrong the probabilities are; it does not say
        whether they are wrong in a consistent direction.  This table does, and
        it is the check a reader should apply before using the bands as
        decision inputs - a surface can hold a respectable skill score and
        still be over-confident in the middle of its range, which is exactly
        the region a decision threshold tends to fall in.

        The bins are equal-count rather than equal-width, so an empty bin means
        the method never produced that probability rather than meaning the bin
        was too narrow to catch anything.
        """
        if cv is None:
            return ""
        rows, worst = calibration_table(
            cv["held_out_predicted"], cv["held_out_observed"], n_bins=bins
        )
        if not rows:
            return ""
        table = [
            (
                f"{low:.2f} to {high:.2f}",
                f"{n:,} held-out score(s), mean predicted {mean:.4f}, observed "
                f"frequency {freq:.4f}, gap {abs(mean - freq):.4f}",
            )
            for low, high, n, mean, freq in rows
        ]
        if worst < 0.05:
            reading = (
                f"Every bin's observed frequency is within {worst:.4f} of the "
                "probability it was predicted at, so the bands are scaled "
                "correctly on this field, not merely ranked correctly. That is "
                "a stronger statement than the skill score above and it is the "
                "one a decision threshold depends on."
            )
        else:
            reading = (
                f"The widest gap between a bin's predicted probability and the "
                f"frequency it actually occurred at is {worst:.4f}. A gap that "
                "size means the bands lean one way over that part of their "
                "range: read them as a ranking rather than as calibrated "
                "probabilities, and do not put a decision threshold inside the "
                "bin where the gap is widest."
            )
        return (
            "<h3>Calibration of the Held-Out Probabilities</h3>"
            "<table>"
            "<thead><tr><th>Predicted Probability Band</th><th>Result</th>"
            f"</tr></thead><tbody>{metric_rows(table)}</tbody></table>"
            f"<p>{html.escape(reading)}</p>"
        )

    def _surface_note(self, variance_path):
        if variance_path:
            return (
                "The prediction, the E-type estimate and its variance were "
                "written as GeoTIFF Float32 rasters covering the extent of the "
                f"input points, with non-finite cells carrying {DEFAULT_NODATA:g}"
                " as no-data. The E-type variance is not the kriging variance: "
                "it is the spread of the value implied by the probability "
                "curve, and the two answer different questions."
            )
        return (
            "The probabilities and the E-type estimate were written as GeoTIFF "
            "Float32 rasters covering the extent of the input points, with "
            f"non-finite cells carrying {DEFAULT_NODATA:g} as no-data. No "
            "destination was given for the E-type variance, so only the summary "
            "statistics above describe it."
        )
