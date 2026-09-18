# -*- coding: utf-8 -*-
"""Interpolation model comparison Processing Algorithm.

Scores several interpolation methods on the same observations and says which
one is worth using on this field.  It is the same question ArcGIS Pro's
*Compare Geostatistical Layers* answers, asked of methods rather than of
finished layers: every method here is driven through the plugin's own
cross-validation protocol, refitting from scratch inside every fold, so the
errors are the errors of the methods and not of their fit to the data they were
trained on.

Two decisions shape everything below and both are about comparability rather
than accuracy.

**One protocol for the whole run.**  A method's error measured by k-fold is
systematically worse than the same method's error measured by leave-one-out,
because each fold's model sees a fraction of the data.  Showing one method's
leave-one-out RMSE beside another's k-fold RMSE would rank the *protocols*, not
the methods.  So the run resolves a single protocol - the strongest one every
selected method can afford - and every method is scored under it.

**Every method refits inside every fold.**  The plugin's kriging tool fits one
variogram on the whole dataset and validates the prediction from it, which is
right for a single kriging run and wrong here: the other methods see no more of
the held-out observation than the fold allows, and a kriging entry that saw it
through the variogram would be the only method scored on a fit it contributed
to.  The variogram is therefore refitted per fold, and the cost of that is why
``KRIGING_REFIT_LOO_LIMIT`` exists.
"""
from __future__ import annotations

import csv
import html
import logging
import math
import os
import tempfile

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterEnum,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
)

from ..core.geostat_engines import (
    MODELS,
    default_max_lag,
    empirical_variogram,
    fit_variogram,
    krige,
)
from ..core.interpolation_engines import (
    IDW_SOFT_LIMIT,
    NATURAL_LOO_LIMIT,
    RBF_LOO_LIMIT,
    TREND_TERM_COUNTS,
    cross_validate,
    cv_protocol_for,
    idw,
    natural_neighbour,
    rbf_predict,
    rbf_weights,
    trend_surface,
)
from ..core.interpolation_report import (
    fmt,
    metric_rows,
    report_foot,
    report_head,
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

#: Observations above which a comparison is refused outright.  Every method is
#: scored by a protocol that refits inside each fold, so the run costs several
#: times what a single interpolation of the same points costs.
COMPARE_SOFT_LIMIT = 3000

#: Observations above which the kriging entry is not scored by leave-one-out.
#: Measured: refitting the variogram and solving once per held-out observation
#: takes 1.1 s at 100 observations, 3.8 s at 300, 9.0 s at 400 and 24.8 s at
#: 600, because each fold factorises an (n-1) x (n-1) system.  Four hundred is
#: where the wait is still a wait rather than a hang; above it the whole run
#: drops to k-fold, which refits once per fold instead of once per observation.
KRIGING_REFIT_LOO_LIMIT = 400

#: Half-width, in standard errors of the paired differences, below which the
#: gap between the leading two methods is called unresolved.  Two standard
#: errors is the conventional line, and stating the number rather than writing
#: "similar" is what stops a ranking being read as a finding it is not.
MARGIN_SIGNIFICANT = 2.0

#: Fewest paired differences worth testing.  Below this the standard error of
#: the difference is itself too poorly determined to judge anything by.
MIN_PAIRED = 8

KRIGING_MODEL_VALUES = ("auto",) + tuple(MODELS)
KRIGING_MODEL_LABELS = ("Automatic - whichever family fits best",) + tuple(
    name.capitalize() for name in MODELS
)


# ── the methods ──────────────────────────────────────────────────────────────


def predict_idw(train_coords, train_values, test_coords, options):
    # The engine returns (prediction, used, exact_hits); the comparison wants
    # only the first.  Returning the tuple lets np.asarray see three ragged
    # items and raise "setting an array element with a sequence", which the
    # caller reports as a method that could not be scored - so the failure is
    # a silently missing method rather than an error, and it took reading the
    # emitted CSV to notice IDW had dropped out of a default run.
    prediction, _used, _exact = idw(train_coords, train_values, test_coords,
                                    power=options["idw_power"])
    return prediction


def predict_natural(train_coords, train_values, test_coords, options):
    prediction, _used, _detail = natural_neighbour(
        train_coords, train_values, test_coords
    )
    return prediction


def predict_rbf(train_coords, train_values, test_coords, options):
    state = rbf_weights(train_coords, train_values, kernel="thin_plate",
                        smoothing=options["rbf_smoothing"])
    return rbf_predict(state, train_coords, test_coords)


def _make_trend_predictor(order):
    def _predict(train_coords, train_values, test_coords, options):
        prediction, _se, _used = trend_surface(
            train_coords, train_values, test_coords, order=order
        )
        return prediction
    return _predict


def _best_fit(lag, semivariance, npairs, model, alpha):
    """The fitted variogram family to krige with, chosen the engine's way.

    ``model="auto"`` tries every family and keeps the lowest Cressie-weighted
    cost, the same rule the semivariogram tool and Empirical Bayesian Kriging
    apply, so "automatically selected" means one thing across the plugin.
    """
    if model != "auto":
        return fit_variogram(lag, semivariance, npairs, model, alpha)
    best = None
    for family in MODELS:
        try:
            candidate = fit_variogram(lag, semivariance, npairs, family, alpha)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate
    if best is None:
        raise ValueError(
            "no semivariogram family could be fitted to this fold's "
            "observations"
        )
    return best


def predict_kriging(train_coords, train_values, test_coords, options):
    """Refit the variogram on the training half, then krige the test points.

    See the module docstring: the refit is what makes this entry comparable
    with the others, and it is the most expensive thing the tool does.
    """
    lag, semivariance, npairs = empirical_variogram(
        train_coords, train_values, n_bins=options["n_bins"],
        max_lag=options["max_lag"],
    )[:3]
    fit = _best_fit(lag, semivariance, npairs, options["kriging_model"],
                    options["alpha"])
    prediction, _variance = krige(
        train_coords, train_values, test_coords, fit["model"], fit["nugget"],
        fit["psill"], fit["range"], drift="none",
        alpha=fit.get("alpha", options["alpha"]),
    )
    return prediction


#: The methods on offer, in the order the parameter lists them.  Each carries
#: the observation count below which it cannot be fitted at all and the count
#: above which it cannot afford leave-one-out; the run's protocol is the
#: strictest of the former and the smallest of the latter.
METHOD_SPECS = (
    {
        "key": "idw",
        "label": "Inverse distance weighting (power 2, every observation)",
        "short": "IDW",
        "predict": predict_idw,
        "loo_limit": IDW_SOFT_LIMIT,
        "min_observations": 4,
        "tool": "Inverse Distance Weighting (IDW) Interpolation",
    },
    {
        "key": "natural",
        "label": "Natural neighbour (Sibson)",
        "short": "Natural neighbour",
        "predict": predict_natural,
        "loo_limit": NATURAL_LOO_LIMIT,
        "min_observations": 4,
        "tool": "Natural Neighbour Interpolation",
    },
    {
        "key": "rbf",
        "label": "Radial basis function (thin plate spline)",
        "short": "RBF (thin plate)",
        "predict": predict_rbf,
        "loo_limit": RBF_LOO_LIMIT,
        "min_observations": 4,
        "tool": "Radial Basis Function (RBF) Interpolation",
    },
    {
        "key": "trend1",
        "label": "Global polynomial trend, order 1 (planar)",
        "short": "Trend, order 1",
        "predict": _make_trend_predictor(1),
        "loo_limit": IDW_SOFT_LIMIT,
        "min_observations": 2 * TREND_TERM_COUNTS[1],
        "tool": "Polynomial Trend Surface",
    },
    {
        "key": "trend2",
        "label": "Global polynomial trend, order 2 (quadratic)",
        "short": "Trend, order 2",
        "predict": _make_trend_predictor(2),
        "loo_limit": IDW_SOFT_LIMIT,
        "min_observations": 2 * TREND_TERM_COUNTS[2],
        "tool": "Polynomial Trend Surface",
    },
    {
        "key": "trend3",
        "label": "Global polynomial trend, order 3 (cubic)",
        "short": "Trend, order 3",
        "predict": _make_trend_predictor(3),
        "loo_limit": IDW_SOFT_LIMIT,
        "min_observations": 2 * TREND_TERM_COUNTS[3],
        "tool": "Polynomial Trend Surface",
    },
    {
        "key": "kriging",
        "label": "Ordinary kriging (variogram refitted inside every fold)",
        "short": "Ordinary kriging",
        "predict": predict_kriging,
        "loo_limit": KRIGING_REFIT_LOO_LIMIT,
        "min_observations": 10,
        "tool": "Ordinary / Universal Kriging Interpolation",
    },
)

METHOD_LABELS = tuple(spec["label"] for spec in METHOD_SPECS)
DEFAULT_METHODS = (0, 1, 2, 6)


def paired_errors(errors_a, errors_b):
    """Compare two methods' absolute errors observation by observation.

    A ranking built from two aggregate RMSEs throws away the fact that both
    methods were scored on exactly the same held-out observations under exactly
    the same split.  That pairing is the most valuable thing the comparison has:
    the difference of the two absolute errors, per observation, cancels the
    part of the difficulty that belongs to the location rather than to the
    method, and leaves a much smaller standard error than either RMSE has.

    Positive differences mean the second method is closer.  Returns ``None``
    when too few observations were scored by both to judge anything, which is a
    real answer to report as "not tested" rather than a number to invent.
    """
    a = np.abs(np.asarray(errors_a, dtype=float))
    b = np.abs(np.asarray(errors_b, dtype=float))
    usable = np.isfinite(a) & np.isfinite(b)
    n = int(usable.sum())
    if n < MIN_PAIRED:
        return None
    differences = a[usable] - b[usable]
    mean = float(differences.mean())
    spread = float(differences.std(ddof=1))
    if spread > 0.0:
        standard_error = spread / math.sqrt(n)
        ratio = abs(mean) / standard_error
    else:
        # Every difference identical: either the two methods agree exactly, or
        # one is better by the same amount everywhere.  Zero spread makes the
        # ratio infinite, which is the honest description of both.
        standard_error = 0.0
        ratio = float("inf") if mean != 0.0 else 0.0
    return {
        "n": n,
        "mean_difference": mean,
        "standard_error": standard_error,
        "ratio": ratio,
        # Fraction of the paired observations on which the second method was
        # strictly closer.  Reported beside the mean because a method can win
        # the average while losing most locations, and those are different
        # findings.
        "second_better_fraction": float((differences > 0.0).mean()),
        "first_better_fraction": float((differences < 0.0).mean()),
        "tied_fraction": float((differences == 0.0).mean()),
    }


def common_observation_mask(results, n):
    """The held-out observations that *every* scored method could predict.

    The intersection is the only subset on which the methods' errors are
    strictly comparable, and the point of taking it is that the observations a
    method cannot predict are not a random sample of them: an interpolator that
    refuses to extrapolate past the convex hull of its training points drops the
    observations at the edge of the extent, which are exactly the ones no method
    finds easy.  Ranking on the per-method subsets would reward that refusal.

    Returns a boolean array of length ``n``; all-True when every method scored
    everything, which is the usual case and costs nothing.
    """
    mask = np.ones(int(n), dtype=bool)
    for row in results:
        mask &= np.isfinite(np.asarray(row["cv"]["errors"], dtype=float))
    return mask


def subset_statistics(cv, mask):
    """The error statistics restricted to ``mask``, computed from the errors.

    The engine's own statistics come from the same arrays, so going through
    them again here keeps one definition of RMSE in the plugin rather than a
    second one that has to agree with the first by inspection.
    """
    errors = np.asarray(cv["errors"], dtype=float)[mask]
    n = int(errors.size)
    if n == 0:
        return {
            "n": 0, "rmse": float("nan"), "mae": float("nan"),
            "me": float("nan"), "max_abs_error": float("nan"), "nrmse": None,
        }
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    std = cv["field_std"]
    return {
        "n": n,
        "rmse": rmse,
        "mae": float(np.mean(np.abs(errors))),
        "me": float(np.mean(errors)),
        "max_abs_error": float(np.max(np.abs(errors))),
        "nrmse": (rmse / std) if std and std > 0 else None,
    }


def comparison_verdict(scored, second, margin):
    """A verdict, a colour, a sentence and a next action for the ranking.

    ``scored`` is the ranked list of methods that produced a score, ``second``
    the runner-up and ``margin`` the paired comparison of the two, or ``None``
    when fewer than two methods could be scored.
    """
    if not scored:
        return (
            "Nothing could be scored", "#718096",
            "No selected method produced a usable cross-validation score on "
            "this layer, so there is no comparison to report.",
            "Check the method list against the observation count, and read the "
            "skipped-method table below for what each one said.",
        )
    if len(scored) == 1:
        leader = scored[0]
        return (
            f"Only {leader['short']} could be scored", "#dd6b20",
            f"One method produced a score ({leader['short']}, RMSE "
            f"{fmt(leader['common']['rmse'])}), so nothing here compares it with "
            "anything. A single number is a measurement, not a comparison.",
            "Select more methods, or add observations - the methods that were "
            "skipped list their own reasons below.",
        )

    leader = scored[0]
    leader_rmse = leader["common"]["rmse"]
    second_rmse = second["common"]["rmse"]
    base = (
        f"On this field {leader['short']} has the lowest held-out error, RMSE "
        f"{fmt(leader_rmse)} against {fmt(second_rmse)} for {second['short']}"
        + (f", a {(1.0 - leader_rmse / second_rmse):.1%} reduction"
           if second_rmse else "")
        + ". "
    )
    if margin is None:
        return (
            f"{leader['short']} leads, margin not tested", "#dd6b20",
            base + "Too few observations were scored by both leading methods "
            "for the gap between them to be judged, so the order of those two "
            "rests on two averages rather than on a test.",
            "Read the ranking as a shortlist rather than an order. The gap "
            f"between {leader['short']} and {second['short']} has not been "
            "shown to exist here.",
        )
    ratio = margin["ratio"]
    if ratio >= MARGIN_SIGNIFICANT:
        return (
            f"{leader['short']} is the best of the methods compared", "#2f855a",
            base + f"The paired difference is {margin['mean_difference']:+.6g} "
            f"with a standard error of {margin['standard_error']:.6g}, so the "
            f"gap is {ratio:.1f} standard errors wide and this sample resolves "
            f"it. {leader['short']} was closer on "
            f"{margin['first_better_fraction']:.1%} of the paired locations "
            f"and {second['short']} on "
            f"{margin['second_better_fraction']:.1%}.",
            f"Use {leader['short']} through the {leader['tool']} tool. Before "
            "committing to it, check the error map it produces: a method can "
            "win on average while failing in one part of the extent.",
        )
    return (
        "The leading methods are not separated by this sample", "#dd6b20",
        base + f"The paired difference is {margin['mean_difference']:+.6g} "
        f"against a standard error of {margin['standard_error']:.6g} - only "
        f"{ratio:.1f} standard errors - and the two methods swap places across "
        f"locations, with {leader['short']} closer on "
        f"{margin['first_better_fraction']:.1%} and {second['short']} on "
        f"{margin['second_better_fraction']:.1%}. A ranking this narrow is not "
        "a finding, it is the noise of this sample.",
        "Choose between them on grounds the data cannot arbitrate: the shape "
        "the method produces between observations, what the field is known to "
        "do, and how the surface will be read. Add observations if the choice "
        "has to be made on evidence.",
    )


class InterpolationComparisonAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    METHODS = "METHODS"
    IDW_POWER = "IDW_POWER"
    RBF_SMOOTHING = "RBF_SMOOTHING"
    KRIGING_MODEL = "KRIGING_MODEL"
    ALPHA = "ALPHA"
    N_BINS = "N_BINS"
    MAX_LAG = "MAX_LAG"
    SEED = "SEED"
    MAX_POINTS = "MAX_POINTS"
    HTML_REPORT = "HTML_REPORT"
    CSV_TABLE = "CSV_TABLE"

    def name(self) -> str:
        return "interp_compare"

    def displayName(self) -> str:
        return "Compare Interpolation Methods"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("interp_compare")

    def createInstance(self):
        return InterpolationComparisonAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Scores several interpolation methods on the same observations and "
            "says which of them is worth using on this field.\n\n"
            "The point of a comparison is that it be fair, and the two things "
            "that make one unfair are handled here explicitly. First, every "
            "method is scored under the *same* cross-validation protocol. "
            "Leave-one-out and k-fold errors are not comparable - a k-fold "
            "model sees only a fraction of the data, so its error is "
            "systematically larger - and a table showing one method's "
            "leave-one-out RMSE beside another's k-fold RMSE ranks the "
            "protocols rather than the methods. So the run picks the strongest "
            "protocol that every selected method can afford and states which "
            "one that is. Second, every method refits from scratch inside every "
            "fold, including kriging, whose variogram is refitted per fold here "
            "even though the kriging tool fits it once. That refitting is the "
            "most expensive thing this tool does.\n\n"
            "Because all the methods are scored on exactly the same held-out "
            "observations under the same split, the errors are paired, and the "
            "comparison uses that. Each pair of methods is compared "
            "observation by observation on the difference of their absolute "
            "errors, which cancels the part of the difficulty that belongs to "
            "the location rather than to the method. The result is reported as "
            "a margin in standard errors, and a margin under two is reported "
            "as unresolved rather than as a winner: a ranking built on two "
            "aggregate averages will happily order methods that this sample "
            "cannot tell apart.\n\n"
            "Methods that cannot be fitted at this observation count are "
            "skipped with their reason stated, never dropped silently, and a "
            "method that fails inside a fold is reported as failed with the "
            "engine's own message.\n\n"
            "The tool compares methods, not finished raster layers - ArcGIS "
            "Pro's Compare Geostatistical Layers does the latter. Comparing "
            "layers tells you which surface you were handed is closer to the "
            "data; comparing methods tells you which method to reach for, which "
            "is the question that has an action attached to it. It also means "
            "the tool writes no surface: run the winner through its own tool, "
            "which the report names, so that the surface is produced with the "
            "settings and the report that tool provides."
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
            QgsProcessingParameterEnum(
                self.METHODS,
                "Methods to compare (each is scored under one shared protocol)",
                options=list(METHOD_LABELS),
                allowMultiple=True,
                defaultValue=list(DEFAULT_METHODS),
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.IDW_POWER,
                "IDW power (applies only if IDW is selected)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=2.0,
                minValue=0.1,
                maxValue=10.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RBF_SMOOTHING,
                "RBF smoothing (0 = an exact interpolator, which is what the "
                "RBF tool uses; applies only if RBF is selected)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.KRIGING_MODEL,
                "Semivariogram family for the kriging entry (refitted inside "
                "every fold)",
                options=list(KRIGING_MODEL_LABELS),
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.ALPHA,
                "Stable model shape exponent for the kriging entry (Stable "
                "only; ignored otherwise)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1.5,
                minValue=0.1,
                maxValue=2.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_BINS,
                "Distance bins for the kriging entry's per-fold variogram",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=15,
                minValue=4,
                maxValue=200,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_LAG,
                "Maximum lag in map units for the kriging entry "
                "(0 = automatic)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SEED,
                "Seed for the k-fold split (fixed so every method sees the "
                "same folds, and so the run is reproducible)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=0,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_POINTS,
                "Maximum observations this comparison will accept",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=COMPARE_SOFT_LIMIT,
                minValue=8,
                maxValue=200_000,
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
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.CSV_TABLE,
                "Comparison table as CSV (optional)",
                fileFilter="CSV files (*.csv)",
                optional=True,
                createByDefault=False,
            )
        )
        self.addOutput(
            QgsProcessingOutputHtml(
                "HTML_REPORT_OUT", "Interpolation comparison report"
            )
        )

    # ── run ──────────────────────────────────────────────────────────────────

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        chosen = self.parameterAsEnums(parameters, self.METHODS, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        max_points = self.parameterAsInt(parameters, self.MAX_POINTS, context)
        n_bins = self.parameterAsInt(parameters, self.N_BINS, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT,
                                               context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(),
                "planx_interpolation_comparison.html",
            )
        csv_path = self.parameterAsFileOutput(parameters, self.CSV_TABLE,
                                              context)

        if not chosen:
            raise QgsProcessingException(
                "No methods were selected. A comparison needs at least two; "
                "select the methods you want scored."
            )
        specs = [METHOD_SPECS[index] for index in sorted(set(chosen))]

        feedback.pushInfo("Reading point locations and target values...")
        read = read_point_values(source, field_name, feedback)
        if read is None:
            return {}
        coords, values, numeric_summary, crs_warning = read
        push_diagnostics(feedback, numeric_summary, None, crs_warning)

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)
        n = len(values_arr)
        require_non_constant(numeric_summary)
        if n > max_points:
            raise QgsProcessingException(
                f"This layer has {n} observations, above the cap of "
                f"{max_points} set for this run. Every method here is scored by "
                "a protocol that refits inside each fold, so the comparison "
                "costs several times what interpolating the same points once "
                "costs. Subset the layer, or raise the cap deliberately if you "
                "accept the wait."
            )

        # ── one protocol for every method ────────────────────────────────────
        # The strongest protocol is the one every selected method can afford,
        # so the strictest limit wins.  Taking the maximum instead would let the
        # cheapest method drag the expensive ones into a stricter test and the
        # table would rank protocols rather than methods.
        runnable = [spec for spec in specs
                    if n >= spec["min_observations"]]
        too_small = [spec for spec in specs
                     if n < spec["min_observations"]]
        if not runnable:
            raise QgsProcessingException(
                f"No selected method can be fitted from {n} observations. "
                + " ".join(
                    f"{spec['short']} needs at least "
                    f"{spec['min_observations']}."
                    for spec in too_small
                )
            )

        joint_limit = min(spec["loo_limit"] for spec in runnable)
        protocol, folds = cv_protocol_for(n, joint_limit)
        binding = min(runnable, key=lambda spec: spec["loo_limit"])
        if protocol == "loo":
            protocol_note = (
                "Every observation was held out in turn and predicted from all "
                "the others, with every model refitted from scratch each time. "
                "Leave-one-out is the strongest test available and the only one "
                "whose result does not depend on a random split."
            )
        else:
            protocol_note = (
                f"The observations were split into {folds} folds and each fold "
                "predicted from the rest, with every model refitted inside "
                "every fold. This is a stricter test than leave-one-out - each "
                "model sees "
                f"{1.0 - 1.0 / folds:.0%} of the data rather than all but one "
                "observation - so every error in the table below is larger than "
                "the same method's leave-one-out figure would be. That is why "
                "the whole table shares one protocol: mixing the two would rank "
                "the protocols rather than the methods."
            )

        feedback.pushInfo(
            f"Comparing {len(runnable)} method(s) over {n} observations by "
            f"{protocol} ({folds} fold(s), seed {seed})."
        )
        # The protocol is stated in the log as well as in the report, because a
        # run's RMSE is unreadable without it and the log is what survives when
        # someone scrolls back to check what was done.
        feedback.pushInfo(protocol_note)
        if protocol == "kfold":
            binding_note = (
                "and this is the tightest limit among the methods selected"
                if len(runnable) > 1 else "and it is the only method selected"
            )
            feedback.pushInfo(
                f"Leave-one-out was not affordable here: {binding['short']} "
                f"caps it at {binding['loo_limit']} observations, "
                f"{binding_note}. Every method was dropped to the same "
                "protocol rather than scored under a better one, so the table "
                "stays comparable."
            )
        if too_small:
            feedback.pushWarning(
                "Skipped for want of observations: "
                + "; ".join(
                    f"{spec['short']} needs {spec['min_observations']}"
                    for spec in too_small
                )
                + "."
            )

        options = {
            "idw_power": self.parameterAsDouble(parameters, self.IDW_POWER,
                                                context),
            "rbf_smoothing": self.parameterAsDouble(
                parameters, self.RBF_SMOOTHING, context
            ),
            "kriging_model": KRIGING_MODEL_VALUES[
                self.parameterAsEnum(parameters, self.KRIGING_MODEL, context)
            ],
            "alpha": self.parameterAsDouble(parameters, self.ALPHA, context),
            "n_bins": n_bins,
            # Resolved once for the whole run, so every fold's variogram is
            # binned over the same distance axis.  Binning each fold over its
            # own axis would make the families chosen in two folds incomparable
            # and the automatic selection not a selection at all.
            "max_lag": default_max_lag(coords_arr)
            if self.parameterAsDouble(parameters, self.MAX_LAG, context) <= 0
            else self.parameterAsDouble(parameters, self.MAX_LAG, context),
        }

        # ── score each method under that one protocol ────────────────────────
        results = []
        skipped = [
            (spec["short"],
             f"needs at least {spec['min_observations']} observations")
            for spec in too_small
        ]
        total_work = len(runnable)

        for index, spec in enumerate(runnable):
            if feedback.isCanceled():
                return {}
            feedback.pushInfo(
                f"[{index + 1}/{total_work}] Scoring {spec['short']} by "
                f"{protocol}..."
            )
            base = 5.0 + 80.0 * (index / max(1, total_work))
            span = 80.0 / max(1, total_work)

            def _predict(train_coords, train_values, test_coords, _spec=spec):
                return _spec["predict"](train_coords, train_values,
                                        test_coords, options)

            def _progress(done, count, _base=base, _span=span):
                feedback.setProgress(int(_base + _span * done / max(1, count)))
                return not feedback.isCanceled()

            try:
                cv = cross_validate(
                    _predict, coords_arr, values_arr, protocol=protocol,
                    folds=folds, seed=seed, progress=_progress,
                )
            except ValueError as exc:
                # A method that cannot be fitted from a fold's training half is
                # skipped with the engine's own words, not dropped and not
                # reported as a bad score: an unfitted method has no score to
                # be bad.
                skipped.append((spec["short"], str(exc)))
                feedback.pushWarning(
                    f"{spec['short']} could not be scored and was skipped: "
                    f"{exc}"
                )
                continue
            except RuntimeError:
                return {}

            results.append({
                "key": spec["key"],
                "short": spec["short"],
                "label": spec["label"],
                "tool": spec["tool"],
                "configuration": self._configuration(spec, options),
                "cv": cv,
            })
            feedback.pushInfo(
                f"    {spec['short']}: RMSE {cv['rmse']:.6g}, MAE "
                f"{cv['mae']:.6g}, ME {cv['me']:+.6g} over "
                f"{cv['n_scored']} scored observation(s)."
            )
            if cv["n_dropped"]:
                feedback.pushWarning(
                    f"{spec['short']} produced no prediction for "
                    f"{cv['n_dropped']} held-out observation(s); they are "
                    "excluded from its statistics and from any paired "
                    "comparison with it."
                )
            if feedback.isCanceled():
                return {}

        if not results:
            raise QgsProcessingException(
                "No selected method could be scored on this layer. "
                + " ".join(f"{name}: {reason}" for name, reason in skipped)
            )

        # ── rank on the observations every method scored ─────────────────────
        # A method that cannot predict some held-out observations - natural
        # neighbour leaves the ones outside a fold's convex hull as NaN - is
        # scored on a smaller and usually easier subset than one that predicts
        # all of them, because the observations it drops are the extreme ones.
        # Ranking on those unequal subsets would let a method win by failing
        # where the field is hardest.  So the statistics that decide the order
        # are recomputed on the common subset, and the unequal figures are kept
        # beside them rather than thrown away.
        common = common_observation_mask(results, n)
        for row in results:
            row["common"] = subset_statistics(row["cv"], common)
        results.sort(key=lambda row: row["common"]["rmse"])
        for rank, row in enumerate(results, start=1):
            row["rank"] = rank
            row["margin"] = None
        leader = results[0]
        for row in results[1:]:
            row["margin"] = paired_errors(leader["cv"]["errors"],
                                          row["cv"]["errors"])

        if common.sum() < n:
            feedback.pushInfo(
                f"Not every method could predict every held-out observation, "
                f"so the ranking is computed on the {int(common.sum())} "
                f"observation(s) that all {len(results)} of them scored. The "
                "per-method counts are in the table."
            )

        second = results[1] if len(results) > 1 else None
        top_margin = second["margin"] if second is not None else None
        verdict, color, verdict_text, next_action = comparison_verdict(
            results, second, top_margin
        )

        if second is not None:
            if top_margin is None:
                feedback.pushInfo(
                    f"{leader['short']} leads {second['short']} on RMSE, but "
                    "too few observations were scored by both for the gap to "
                    "be tested."
                )
            else:
                feedback.pushInfo(
                    f"{leader['short']} against {second['short']}: mean "
                    f"difference in absolute error "
                    f"{top_margin['mean_difference']:+.6g} with a standard "
                    f"error of {top_margin['standard_error']:.6g}, "
                    f"{top_margin['ratio']:.1f} standard errors."
                )

        if csv_path:
            feedback.pushInfo("Writing the comparison table as CSV...")
            self._write_csv(csv_path, results, field_name, n, protocol, folds,
                            seed, numeric_summary)

        feedback.setProgress(95)
        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, results, skipped, protocol, folds, seed,
            binding, too_small, numeric_summary, crs_warning, options,
            verdict, color, verdict_text, next_action, source,
        )

        result = {
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }
        if csv_path:
            result[self.CSV_TABLE] = csv_path
        feedback.setProgress(100)
        return result

    # ── supporting output ────────────────────────────────────────────────────

    def _configuration(self, spec, options):
        """The one-line description of how a method was actually configured."""
        if spec["key"] == "idw":
            return f"power {options['idw_power']:g}, global search"
        if spec["key"] == "natural":
            return "Sibson weights over the Delaunay mesh"
        if spec["key"] == "rbf":
            smoothing = options["rbf_smoothing"]
            return (
                "thin plate spline, exact"
                if smoothing == 0.0 else
                f"thin plate spline, smoothing {smoothing:g}"
            )
        if spec["key"].startswith("trend"):
            order = int(spec["key"][-1])
            return (f"order {order}, {TREND_TERM_COUNTS[order]} coefficients, "
                    "a single global fit")
        if spec["key"] == "kriging":
            family = KRIGING_MODEL_LABELS[
                KRIGING_MODEL_VALUES.index(options["kriging_model"])
            ]
            return (
                f"ordinary kriging, {family.lower()}, {options['n_bins']} bins, "
                f"max lag {options['max_lag']:.6g}, variogram refitted per fold"
            )
        return ""

    def _write_csv(self, path, results, field_name, n, protocol, folds, seed,
                   numeric_summary):
        """The comparison table as CSV, with the run's protocol in the header.

        The protocol is written into the file rather than left to the report
        because a CSV outlives the report it came from, and an RMSE without the
        protocol it was measured under is not a number anyone can use.
        """
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                f"# Interpolation method comparison, field '{field_name}'\n"
            )
            handle.write(f"# observations,{n}\n")
            handle.write(f"# protocol,{protocol}\n")
            handle.write(f"# folds,{folds}\n")
            handle.write(f"# split seed,{seed}\n")
            handle.write(
                "# every method refits from scratch inside every fold\n"
            )
            handle.write(
                "# the error columns are computed over the observations every "
                "scored method could predict,\n"
                "# which is observations_compared, not observations_scored\n"
            )
            handle.write(
                "# margin_vs_leader is the leader's absolute error minus this "
                "method's, so a positive\n"
                "# margin means this method is closer than the leader and a "
                "negative one further away\n"
            )
            writer = csv.writer(handle)
            writer.writerow([
                "rank", "method", "configuration", "protocol",
                "observations_compared", "observations_scored",
                "observations_dropped",
                "mean_error", "rmse", "nrmse", "mae", "max_abs_error",
                "margin_vs_leader", "margin_standard_error",
                "margin_in_standard_errors",
            ])
            for row in results:
                cv = row["cv"]
                stats = row["common"]
                margin = row.get("margin")
                writer.writerow([
                    row["rank"], row["short"], row["configuration"],
                    cv["protocol"], stats["n"], cv["n_scored"],
                    cv["n_dropped"],
                    f"{stats['me']:.10g}", f"{stats['rmse']:.10g}",
                    "" if stats["nrmse"] is None else f"{stats['nrmse']:.10g}",
                    f"{stats['mae']:.10g}", f"{stats['max_abs_error']:.10g}",
                    "" if margin is None else f"{margin['mean_difference']:.10g}",
                    "" if margin is None else f"{margin['standard_error']:.10g}",
                    "" if margin is None else f"{margin['ratio']:.10g}",
                ])

    # ── report ───────────────────────────────────────────────────────────────

    def write_html_report(self, path, field_name, n, results, skipped, protocol,
                          folds, seed, binding, too_small, numeric_summary,
                          crs_warning, options, verdict, color, verdict_text,
                          next_action, sources):
        leader = results[0]
        # Every row carries the same common mask, so the first row's count is
        # the run's common count.
        common_n = results[0]["common"]["n"]

        settings_pairs = [
            ("Observations", f"{n:,}"),
            ("Observations compared",
             f"{common_n:,}"
             + (" (every method scored every observation)"
                if common_n == n else
                " (the ones every scored method could predict)")),
            ("Methods compared", f"{len(results)}"),
            ("Validation protocol",
             "leave-one-out, all but one observation in every fit"
             if protocol == "loo" else
             f"{folds}-fold, refitting inside every fold"),
            ("Split seed", f"{seed} (identical for every method, so the "
                           "errors are paired)"),
            ("Protocol set by",
             f"{binding['short']}, the most constrained method selected "
             f"(limit {binding['loo_limit']:,} observations)"
             if len(results) + len(too_small) > 1 else
             f"{binding['short']}, the only method selected"),
            ("IDW power", f"{options['idw_power']:g}"),
            ("RBF smoothing", f"{options['rbf_smoothing']:g}"),
            ("Kriging variogram family",
             KRIGING_MODEL_LABELS[
                 KRIGING_MODEL_VALUES.index(options["kriging_model"])
             ]),
            ("Kriging maximum lag",
             f"{options['max_lag']:.6g} map units"),
        ]

        table_rows = []
        for row in results:
            cv = row["cv"]
            stats = row["common"]
            margin = row.get("margin")
            if margin is None:
                margin_text = "&mdash; (leader)" if row["rank"] == 1 else "n/a"
            else:
                margin_text = (
                    f"{margin['mean_difference']:+.6g} &plusmn; "
                    f"{margin['standard_error']:.6g} "
                    f"({margin['ratio']:.1f} SE)"
                )
            # A method that could not predict everything shows how many it
            # managed, so the reader can see why the ranking is computed on a
            # narrower subset than the run scored.  The dagger marks the other
            # half of the same problem: a method that scored *more* than the
            # common subset has observations excluded from the columns beside
            # it, and without the mark its own count would look inconsistent
            # with the RMSE it is printed next to.
            if cv["n_scored"] == cv["n"]:
                scored_text = f"{cv['n_scored']:,}"
            else:
                scored_text = f"{cv['n_scored']:,} of {cv['n']:,}"
            if cv["n_scored"] > stats["n"]:
                scored_text += " &dagger;"
            table_rows.append(
                "<tr>"
                f'<td class="metric-name">{row["rank"]}</td>'
                f'<td class="metric-name">{html.escape(row["short"])}</td>'
                f"<td>{scored_text}</td>"
                f"<td>{fmt(stats['rmse'])}</td>"
                f"<td>{'n/a' if stats['nrmse'] is None else fmt(stats['nrmse'])}</td>"
                f"<td>{fmt(stats['mae'])}</td>"
                f"<td>{fmt(stats['me'])}</td>"
                f"<td>{fmt(stats['max_abs_error'])}</td>"
                f"<td>{margin_text}</td>"
                "</tr>"
            )

        configuration_pairs = [
            (row["short"], row["configuration"]) for row in results
        ]
        if skipped:
            configuration_pairs += [
                (f"{name} - not scored", reason) for name, reason in skipped
            ]

        margin_html = self._margin_section(results)
        comparison_note = self._comparison_note(results, protocol, folds,
                                               leader, numeric_summary)

        method_note = (
            "Each method was refitted from scratch inside every validation "
            "fold and predicted the observations held out from that fit. This "
            "is the only way to score an exact interpolator: IDW, a thin-plate "
            "spline and natural neighbour all pass through their own "
            "observations, so a residual measured against a surface fitted to "
            "the same data is zero by construction and says nothing about the "
            "method. The numbers below are therefore errors where the method "
            "had not been."
        )

        kriging_note = (
            "One entry does more work than the others and it is worth knowing "
            "which. The kriging tool fits a single variogram on the whole "
            "dataset and validates the prediction from it; here the variogram "
            "is refitted inside every fold, because otherwise kriging would be "
            "the one method whose fit had already seen the observation it was "
            "being scored on. That refit is why the comparison caps how many "
            "observations it will score by leave-one-out. It also means the "
            "kriging RMSE in this table is not the number the kriging tool "
            "reports for the same field, and the difference is the refit, not "
            "a disagreement."
        )

        guidance_html = analyst_guidance_html(
            "Interpolation Method Comparison",
            "Scores several interpolation methods on the same observations "
            "under one shared cross-validation protocol and reports which is "
            "worth using on this field.",
            [
                "All the methods that were scored used the same protocol and "
                "the same split, so the table compares methods and not tests.",
                "The margin between the leading two is at least two standard "
                "errors of the paired differences, so the order means "
                "something.",
                "The winning method's error is not concentrated in one part of "
                "the extent - check by interpolating with the tool named below "
                "and reading its residual map.",
            ],
            [
                "Treating the ranking as resolved when the margin is under two "
                "standard errors. The tool says so where that happens; a short "
                "table of numbers invites a reader to ignore it.",
                "Comparing these RMSEs with the figure the individual tool "
                "reports for the same method. The protocol and the refit "
                "policy have to match before two errors mean the same thing, "
                "and here they do not always.",
                "Choosing a method on the ranking alone when two of them are "
                "level. Where the data cannot separate them, the shape the "
                "method produces between observations is the real difference.",
                "Adding a method to the list and reading the old ranking "
                "against the new one. The protocol is resolved from the "
                "methods selected, so changing the list can change the test.",
            ],
            [
                f"Run {leader['short']} through the {leader['tool']} tool to "
                "produce the surface, with the settings this report records",
                "Empirical Semivariogram & Model Fitting to see whether the "
                "field has the structure kriging can use at all",
                "The threshold exceedance and probability tools if the "
                "question is a probability rather than a value",
            ],
            "A comparison answers which method to reach for, not what the "
            "field is. Run the winner and read its own report: the surface, "
            "its uncertainty and its caveats live there, and this table has "
            "none of them.",
        )

        caveat_text = [
            "The protocol is the strongest one *every selected method* could "
            "afford, so adding a method with a low limit can make the whole "
            "table stricter and every error larger. The settings table above "
            "names which method set that limit.",
            "A k-fold figure is systematically worse than a leave-one-out "
            "figure for the same method on the same data, because each fold's "
            "model sees a fraction of the observations. The table never mixes "
            "the two, so the numbers are comparable with each other and with "
            "nothing outside the run.",
            "The margin test compares absolute errors observation by "
            "observation on the held-out set. It is a statement about this "
            "sample at these locations: a margin that is resolved here can "
            "disappear over different ground, and a margin that is not "
            "resolved here is not evidence that the methods are equivalent "
            "anywhere else.",
            "The margin test treats the paired differences as independent "
            "across observations, and they are not. Two held-out observations "
            "close together share most of their training data and the same "
            "local terrain, so their differences move together, and a field "
            "with strong spatial structure has far fewer independent "
            "comparisons in it than it has observations. The standard error "
            "reported above is therefore smaller than the evidence warrants "
            "and the margin in standard errors is the optimistic one: a margin "
            "that clears two on a scatter of close-together points is weaker "
            "than the same margin spread evenly over the extent, and nothing "
            "in this table tells those two apart.",
            "The error statistics say nothing about the shape of the surface "
            "between observations. Two methods can score almost identically "
            "and produce visibly different maps - one smooth, one with "
            "bull's-eyes around the observations - and the choice between them "
            "is then a choice about the map, not about the error.",
            "Observations that a method could not predict - a natural-neighbour "
            "target whose geometry is degenerate, a trend order that a fold's "
            "training half cannot support - are excluded from that method's "
            "statistics and from any paired comparison with it, and the table "
            "reports how many.",
        ]

        summary_txt = (
            f"{len(results)} interpolation method(s) were scored on "
            f"{n:,} observations from <strong>{html.escape(field_name)}"
            f"</strong> under one shared protocol. <strong>"
            f"{html.escape(leader['short'])}</strong> has the lowest held-out "
            f"error at RMSE {fmt(leader['common']['rmse'])}."
        )
        if len(results) > 1:
            margin = results[1].get("margin")
            if margin is None:
                summary_txt += (
                    " The gap to the runner-up could not be tested on this "
                    "sample."
                )
            elif margin["ratio"] >= MARGIN_SIGNIFICANT:
                summary_txt += (
                    f" It leads {html.escape(results[1]['short'])} by "
                    f"{fmt(abs(margin['mean_difference']))} in mean absolute "
                    f"error, {margin['ratio']:.1f} standard errors of the "
                    "paired differences."
                )
            else:
                summary_txt += (
                    f" The gap to {html.escape(results[1]['short'])} is only "
                    f"{margin['ratio']:.1f} standard errors wide, so this "
                    "sample does not separate them."
                )

        head = report_head(
            "Interpolation Method Comparison",
            f"Field Compared: <strong>{html.escape(field_name)}</strong> | "
            f"Observations: <strong>{n:,}</strong> | Methods Scored: "
            f"<strong>{len(results)}</strong>",
            verdict, verdict_text, color,
        )

        body = f"""
    <section>
        <h2>Executive Summary</h2>
        <p>{summary_txt}</p>
    </section>

    <section>
        <h2>How the Comparison Was Run</h2>
        <table>
            <thead><tr><th>Setting</th><th>Value</th></tr></thead>
            <tbody>{metric_rows(settings_pairs)}</tbody>
        </table>
        <p>{html.escape(method_note)}</p>
        <p>{html.escape(protocol_note_text(protocol, folds))}</p>
        <p>{html.escape(kriging_note)}</p>
    </section>

    <section>
        <h2>The Comparison Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th><th>Method</th><th>Scored</th><th>RMSE</th>
                    <th>NRMSE</th><th>MAE</th><th>ME</th><th>Max Error</th>
                    <th>Margin vs Leader</th>
                </tr>
            </thead>
            <tbody>{''.join(table_rows)}</tbody>
        </table>
        <p>The margin column reports the mean difference in absolute error
        against the leader, with its standard error and that difference
        expressed in standard errors. NRMSE divides the RMSE by the field's
        standard deviation, so it is the one column comparable with a
        comparison run on a different field.</p>
        <p>Every error statistic in this table is computed over the
        <strong>{common_n:,}</strong> observation(s) that <em>all</em> the
        scored methods could predict. That is deliberate: an interpolator that
        declines to extrapolate past the edge of its training points - natural
        neighbour does, which is why it may show fewer scored observations -
        would otherwise be measured on a smaller and easier subset than a
        method that answers everywhere, and could win by failing where the
        field is hardest. The Scored column shows how many each method managed
        on its own; where it is smaller than the common count, the observations
        it dropped are excluded from the comparison rather than counted as
        errors. A count marked &dagger; scored <em>more</em> than that common
        subset - the extra observations are ones no other method could predict,
        so they are excluded from the error columns beside it.</p>
    </section>

    <section>
        <h2>Method Configurations</h2>
        <table>
            <thead><tr><th>Method</th><th>Configuration</th></tr></thead>
            <tbody>{metric_rows(configuration_pairs)}</tbody>
        </table>
    </section>

    {margin_html}

    <section>
        <h2>Reading the Ranking</h2>
        <p>{html.escape(comparison_note)}</p>
    </section>

    {diagnostics_html(numeric_summary, None, crs_warning)}

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + body + report_foot(
                caveats_html("Interpolation Method Comparison", None,
                             numeric_summary, caveat_text),
                guidance_html,
            ))

    def _margin_section(self, results):
        """The paired comparisons, one row per method against the leader."""
        rows = []
        for row in results[1:]:
            margin = row.get("margin")
            if margin is None:
                rows.append((
                    f"{row['short']} against the leader",
                    "not tested - fewer than "
                    f"{MIN_PAIRED} observations were scored by both methods",
                ))
                continue
            direction = (
                "closer than the leader" if margin["mean_difference"] > 0
                else "further than the leader"
            )
            rows.append((
                f"{row['short']} against the leader",
                f"mean difference in absolute error "
                f"{margin['mean_difference']:+.6g}, so this method is "
                f"{direction} by {fmt(abs(margin['mean_difference']))} on "
                f"average; standard error {margin['standard_error']:.6g}, "
                f"margin {margin['ratio']:.1f} standard errors wide over "
                f"{margin['n']:,} paired observation(s). It was closer on "
                f"{margin['second_better_fraction']:.1%} of them, the leader "
                f"on {margin['first_better_fraction']:.1%}"
                + (f", and the two agreed exactly on "
                   f"{margin['tied_fraction']:.1%}."
                   if margin["tied_fraction"] > 0 else ".")
            ))
        if not rows:
            return ""
        reading = (
            "The paired test is the reason this section exists rather than a "
            "sentence about two RMSEs. Because every method was scored on "
            "exactly the same held-out observations under the same split, the "
            "difference in their absolute errors can be taken location by "
            "location, which removes the part of the difficulty that belongs "
            "to where the observation is rather than to the method being "
            f"judged. A margin under {MARGIN_SIGNIFICANT:g} standard errors is "
            "reported here as unresolved: it is an ordering that rests on two "
            "averages rather than on evidence from this sample."
        )
        return (
            "<section>"
            "<h2>Is the Ranking Real? Paired Comparisons</h2>"
            "<table>"
            "<thead><tr><th>Comparison</th><th>Result</th></tr></thead>"
            f"<tbody>{metric_rows(rows)}</tbody></table>"
            f"<p>{html.escape(reading)}</p>"
            "</section>"
        )

    def _comparison_note(self, results, protocol, folds, leader, summary):
        """The prose on what the ranking does and does not establish."""
        parts = [
            f"RMSE is in the field's own units, so the ranking is in those "
            f"units too; the NRMSE column divides by the field's standard "
            f"deviation ({fmt(summary['std'])}) and is the column to compare "
            "with a run on different data.",
            "The comparison was run by "
            + ("leave-one-out" if protocol == "loo" else f"{folds}-fold")
            + " cross-validation, and every method in the table was scored "
            "under it. Had the methods been scored under different protocols "
            "the ordering would be partly a statement about the tests.",
        ]
        if len(results) > 1:
            margin = results[1].get("margin")
            if margin is not None and margin["ratio"] < MARGIN_SIGNIFICANT:
                parts.append(
                    "The leader and the runner-up are level on this sample, "
                    "and the rest of the table should be read as a shortlist "
                    "rather than an order. Where the data does not separate "
                    "methods, the honest basis for choosing is the shape each "
                    "one produces: inverse distance and a thin-plate spline "
                    "will ring around sparse observations, a polynomial will "
                    "smooth through them, and kriging will do neither."
                )
            elif margin is not None:
                parts.append(
                    f"{leader['short']} beats the runner-up by more than the "
                    "noise of this sample, so the ranking is worth acting on. "
                    "It is still a statement about these locations: check the "
                    "winner's own report for where its error is concentrated."
                )
        else:
            parts.append(
                "Only one method could be scored, so this table is a "
                "measurement rather than a comparison and no ranking is being "
                "claimed."
            )
        return " ".join(parts)


def protocol_note_text(protocol, folds):
    """The protocol paragraph, shared by the report and the log."""
    if protocol == "loo":
        return (
            "Every observation was held out in turn and predicted from all the "
            "others, with every model refitted from scratch on each occasion. "
            "This is the strongest test the data allows and the only one that "
            "does not depend on a random split, so no seed can change it."
        )
    return (
        f"The observations were split into {folds} folds by a fixed seed and "
        "each fold predicted from the rest, with every model refitted inside "
        f"every fold. Every method saw the same {folds} folds, which is what "
        "makes their errors paired and the margin test between them valid. "
        f"Note that a {folds}-fold error is systematically larger than the "
        "same method's leave-one-out error would be, because each model is "
        "fitted on a fraction of the observations: the figure is not "
        "comparable with a leave-one-out figure from another run."
    )
