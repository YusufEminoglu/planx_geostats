# -*- coding: utf-8 -*-
"""Empirical Semivariogram & Model Fitting Processing Algorithm."""
from __future__ import annotations

import logging
import os
import tempfile
import html
import numpy as np

from qgis.PyQt.QtCore import QVariant
from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsFeature,
    QgsFeatureSink,
    QgsFields,
    QgsField,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFileDestination,
    QgsWkbTypes,
)

from ..core.geostat_engines import (
    MODELS,
    PARAM_LAYER_FIELDS,
    anisotropy_rose,
    default_max_lag,
    directional_variogram,
    empirical_variogram,
    fit_variogram,
    fitted_curve,
    quality_flags,
)
from ..core.analysis_diagnostics import (
    caveats_html,
    crs_unit_warning,
    diagnostics_html,
    numeric_quality_summary,
    push_diagnostics,
)
from ..core.layer_metadata import apply_output_metadata
from ..core.reporting import analyst_guidance_css, analyst_guidance_html
from ..core.charts import (
    chart_css,
    semivariogram_svg,
    variogram_rose_svg,
)
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon
from ._deliver import style_loaded_output


logger = logging.getLogger("PlanX GeoStats Lab")

MODEL_LABELS = ("Spherical", "Exponential", "Gaussian", "Linear", "Stable")

#: Type of each fitted-parameter attribute, keyed by the field name.  The
#: *names* and their order come from core.geostat_engines.PARAM_LAYER_FIELDS,
#: which the kriging tool imports too; this dict only adds the Qt types.  A
#: name here that the engine does not list (or vice versa) raises immediately
#: when the output fields are built, instead of producing a parameter layer the
#: kriging tool cannot read.
_FIELD_TYPES = {
    "model": (QVariant.String, 20, 0),
    "nugget": (QVariant.Double, 20, 8),
    "psill": (QVariant.Double, 20, 8),
    "sill": (QVariant.Double, 20, 8),
    "range": (QVariant.Double, 20, 8),
    "alpha": (QVariant.Double, 20, 8),
    "r2": (QVariant.Double, 20, 8),
    "rmse": (QVariant.Double, 20, 8),
    "bins": (QVariant.Int, 0, 0),
    "npoints": (QVariant.Int, 0, 0),
    "dir_ratio": (QVariant.Double, 20, 8),
}


def parameter_output_fields() -> QgsFields:
    """Build the parameter layer's fields from the shared name contract."""
    fields = QgsFields()
    for name in PARAM_LAYER_FIELDS:
        variant, length, precision = _FIELD_TYPES[name]
        if variant == QVariant.Int:
            fields.append(QgsField(name, variant))
        else:
            fields.append(QgsField(name, variant, len=length, prec=precision))
    return fields


class EmpiricalSemivariogramAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    N_BINS = "N_BINS"
    MAX_LAG = "MAX_LAG"
    MODEL = "MODEL"
    ALPHA = "ALPHA"
    N_DIRECTIONS = "N_DIRECTIONS"
    DIRECTION_TOLERANCE = "DIRECTION_TOLERANCE"
    PARAMS_OUT = "PARAMS_OUT"
    HTML_REPORT = "HTML_REPORT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "empirical_semivariogram"

    def displayName(self) -> str:
        return "Empirical Semivariogram & Model Fitting"

    def group(self) -> str:
        return "07 | Interpolation and Geostatistics"

    def groupId(self) -> str:
        return "planx_interpolation_geostatistics"

    def icon(self):
        return algorithm_icon("empirical_semivariogram")

    def createInstance(self):
        return EmpiricalSemivariogramAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes the empirical semivariogram of a numeric field and fits a "
            "theoretical model to it - the first half of the classical "
            "geostatistics workflow, and the input the kriging tool needs.\n\n"
            "The empirical semivariogram estimates how quickly values stop "
            "resembling each other as the distance between them grows: for "
            "every pair of points it takes half the squared difference of "
            "their values, then averages those half-differences within "
            "distance bins. A field with spatial structure rises from near "
            "zero and levels off at the variance of the field. Three fitted "
            "parameters describe that shape: the nugget (the discontinuity at "
            "zero distance, i.e. variation below the sampling resolution plus "
            "measurement error), the sill (the level the curve reaches, "
            "approximating the field's total variance), and the range (the "
            "distance beyond which two locations are effectively unrelated).\n\n"
            "Five models are offered: spherical, exponential, gaussian, linear, "
            "and stable. The stable model adds a shape exponent (Alpha below) "
            "that the other four do not use - Alpha is ignored unless Stable is "
            "selected. Fitting minimises the Cressie (1985) weighted sum of "
            "squares, which downweights the long-lag bins that rest on few "
            "point pairs and are the noisiest part of the estimate. The "
            "reported R2 describes how closely the fitted curve follows the "
            "binned points.\n\n"
            "Because the semivariogram is direction-independent here, the "
            "fitted model is isotropic. Directional empirical variograms and a "
            "variogram rose are also reported, so anisotropy can be seen when "
            "it is present - but it is a diagnostic, not something the fitted "
            "model absorbs, and kriging built from these parameters will be "
            "isotropic.\n\n"
            "Two cautions worth reading before trusting the numbers. The range "
            "is weakly identified from a finite sample: a curve can track the "
            "binned points closely (R2 above 0.9) while the fitted range is "
            "20-30 per cent away from the true one, because nugget, sill and "
            "range trade off against each other. And distances are in the "
            "layer's map units, so a geographic CRS makes them degrees unless "
            "the layer is projected first.\n\n"
            "Output: an HTML report with the fitted parameters, a plot of the "
            "empirical points against the fitted curve, and the directional "
            "diagnostics; plus a single-feature parameter layer that the "
            "ordinary kriging tool reads directly, so the fitted model does "
            "not have to be retyped between the two dialogs."
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
                "Target numeric field to model",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_BINS,
                "Number of distance bins",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=15,
                minValue=4,
                maxValue=200,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_LAG,
                "Maximum lag distance (map units; 0 = automatic)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODEL,
                "Theoretical semivariogram model",
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
                self.N_DIRECTIONS,
                "Directional variogram directions (0 = skip the rose)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=8,
                minValue=0,
                maxValue=24,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DIRECTION_TOLERANCE,
                "Directional tolerance (degrees, half-angle of each cone)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=22.5,
                minValue=5.0,
                maxValue=90.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.PARAMS_OUT,
                "Fitted semivariogram parameters (for the kriging tool)",
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
                "HTML_REPORT_OUT", "Semivariogram model fitting report"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        n_bins = self.parameterAsInt(parameters, self.N_BINS, context)
        max_lag = self.parameterAsDouble(parameters, self.MAX_LAG, context)
        model_idx = self.parameterAsEnum(parameters, self.MODEL, context)
        model = MODELS[model_idx]
        alpha = self.parameterAsDouble(parameters, self.ALPHA, context)
        n_directions = self.parameterAsInt(parameters, self.N_DIRECTIONS, context)
        tolerance = self.parameterAsDouble(
            parameters, self.DIRECTION_TOLERANCE, context
        )

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(
                tempfile.gettempdir(), "planx_semivariogram_report.html"
            )

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        feedback.pushInfo("Reading point locations and target values...")
        coords = []
        values = []
        y_dict = {}
        missing_geometry = 0
        for f in source.getFeatures():
            if feedback.isCanceled():
                return {}
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
            y_dict[f.id()] = z

        numeric_summary = numeric_quality_summary(source.featureCount(), y_dict, values)
        crs_warning = crs_unit_warning(source)
        push_diagnostics(feedback, numeric_summary, None, crs_warning)
        if missing_geometry:
            feedback.pushWarning(
                f"{missing_geometry} feature(s) had no usable centroid and were "
                "excluded from the variogram."
            )

        n = len(values)
        if n < 10:
            raise QgsProcessingException(
                "At least 10 features with a valid geometry and a finite numeric "
                f"value are required to estimate a semivariogram; got {n}."
            )
        if numeric_summary["is_constant"]:
            raise QgsProcessingException(
                "The target field is constant, so every pair difference is zero "
                "and there is no spatial structure to model."
            )

        coords_arr = np.asarray(coords, dtype=float)
        values_arr = np.asarray(values, dtype=float)

        auto_lag = default_max_lag(coords_arr)
        lag_limit = float(max_lag) if max_lag and max_lag > 0 else float(auto_lag)
        if not np.isfinite(lag_limit) or lag_limit <= 0:
            raise QgsProcessingException(
                "Every input feature shares the same location, so no pairwise "
                "distances can be formed."
            )

        feedback.pushInfo(
            f"Binning {n * (n - 1) // 2} point pairs into {n_bins} distance bins "
            f"up to {lag_limit:.6g} map units..."
        )
        lag, semivariance, npairs, _dist, _sv = empirical_variogram(
            coords_arr, values_arr, n_bins=n_bins, max_lag=lag_limit
        )
        if len(lag) < 4:
            raise QgsProcessingException(
                "Only "
                f"{len(lag)} distance bin(s) contained point pairs, but a "
                "semivariogram model needs at least 4 to be fitted. Increase the "
                "number of distance bins, or widen the maximum lag distance."
            )

        feedback.pushInfo(f"Fitting the {model} semivariogram model...")
        try:
            fitted = fit_variogram(
                lag, semivariance, npairs, model=model, alpha=alpha
            )
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        feedback.pushInfo(
            f"Fitted {model}: nugget={fitted['nugget']:.6g}, "
            f"sill={fitted['sill']:.6g}, range={fitted['range']:.6g}, "
            f"R2={fitted['r2']:.4f}"
        )

        rose = None
        directions = None
        if n_directions >= 4:
            feedback.pushInfo(
                f"Fitting directional variograms in {n_directions} directions..."
            )
            rose = anisotropy_rose(
                coords_arr, values_arr,
                n_directions=n_directions,
                tolerance_deg=tolerance,
                n_bins=max(4, n_bins // 2),
                max_lag=lag_limit,
            )
            directions = [
                directional_variogram(
                    coords_arr, values_arr, ang, tolerance,
                    n_bins=max(4, n_bins // 2), max_lag=lag_limit,
                )
                for ang in rose["angles_deg"]
            ]
        else:
            feedback.pushInfo(
                "Directional variograms skipped (directions set to 0). "
                "Anisotropy cannot be assessed from this run."
            )

        if feedback.isCanceled():
            return {}

        # The lag limit has to reach quality_flags: without it the fitted range
        # is never compared with the largest distance binned, and the one warning
        # this tool most needs - a variogram that was still rising at the last
        # bin, so the range describes the data's extent rather than the field -
        # can never be raised.
        flags = quality_flags(fitted, max_lag=lag_limit)

        feedback.pushInfo("Writing the fitted-parameter layer...")
        out_fields = parameter_output_fields()

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.PARAMS_OUT,
            context,
            out_fields,
            QgsWkbTypes.Point,
            source.sourceCrs(),
        )
        if sink is None:
            raise QgsProcessingException(
                "Could not create the fitted-parameter output layer."
            )
        self.out_layer_id = dest_id

        feat = QgsFeature(out_fields)
        feat.setGeometry(_single_point(coords_arr))
        ratio = rose["max_min_ratio"] if rose else float("nan")
        feat.setAttributes([
            model,
            float(fitted["nugget"]),
            float(fitted["psill"]),
            float(fitted["sill"]),
            float(fitted["range"]),
            float(alpha),
            float(fitted["r2"]) if np.isfinite(fitted["r2"]) else None,
            float(fitted["rmse"]),
            int(fitted["bins_used"]),
            int(n),
            float(ratio) if np.isfinite(ratio) else None,
        ])
        sink.addFeature(feat, QgsFeatureSink.FastInsert)
        del sink

        feedback.pushInfo("Generating HTML report...")
        self.write_html_report(
            html_path, field_name, n, model, alpha, fitted, lag, semivariance,
            npairs, flags, rose, directions, numeric_summary, crs_warning,
            lag_limit, int(max_lag > 0),
        )

        return {
            self.PARAMS_OUT: dest_id,
            self.HTML_REPORT: html_path,
            "HTML_REPORT_OUT": html_path,
        }

    def write_html_report(
        self, path, field_name, n, model, alpha, fitted, lag, semivariance,
        npairs, flags, rose, directions, numeric_summary, crs_warning,
        lag_limit, lag_was_manual,
    ):
        r2 = fitted["r2"]
        n_ratio = (fitted["nugget"] / fitted["sill"]
                   if fitted["sill"] > 0 else float("nan"))

        if np.isfinite(r2) and r2 >= 0.8:
            verdict, color = "Well-supported fit", "#2f855a"
            desc = (
                f"The fitted {model} model follows the binned semivariance "
                f"closely (R2 = {r2:.3f}). The nugget, sill and range below are "
                "a reasonable summary of this field's spatial structure."
            )
        elif np.isfinite(r2) and r2 >= 0.5:
            verdict, color = "Moderate fit", "#dd6b20"
            desc = (
                f"The fitted {model} model follows the binned semivariance only "
                f"partly (R2 = {r2:.3f}). Treat the parameters as indicative, and "
                "consider trying another model before kriging with them."
            )
        else:
            verdict, color = "Weak fit", "#c53030"
            desc = (
                f"The fitted {model} model explains little of the binned "
                f"semivariance (R2 = {r2:.3f}). This usually means the field has "
                "no spatial structure at this sampling scale, or the distance "
                "bins are too few or too wide."
            )

        h_fit, g_fit = fitted_curve(
            model, fitted["nugget"], fitted["psill"], fitted["range"],
            float(lag_limit), n=140, alpha=alpha,
        )
        plot = semivariogram_svg(
            list(lag), list(semivariance),
            fitted=(list(h_fit), list(g_fit)),
            npairs=list(npairs),
            nugget=fitted["nugget"], sill=fitted["sill"],
            range_=fitted["range"], model_label=model,
            title="Empirical semivariogram and fitted model",
        )

        if rose is not None:
            rose_svg = variogram_rose_svg(
                list(rose["ranges"]), list(rose["angles_deg"]),
                title="Directional range by orientation (variogram rose)",
            )
            n_fitted_dirs = int(np.isfinite(rose["ranges"]).sum())
            rose_model_note = (
                " Each direction is fitted with the spherical model so that the "
                "directions are comparable with one another; this is separate "
                "from the model selected above, and the fitted model itself "
                "remains isotropic either way."
            )
            if np.isfinite(rose["max_min_ratio"]) and rose["max_min_ratio"] > 2.0:
                aniso_note = (
                    f"The fitted range varies by a factor of "
                    f"{rose['max_min_ratio']:.2f} between the longest and "
                    "shortest direction. Read this as a prompt to inspect the "
                    "rose rather than as proof of anisotropy: at the pair counts "
                    "available per direction, scatter of this size also occurs "
                    "in fields that are genuinely isotropic, and this ratio has "
                    "no significance test attached to it. What would be "
                    "convincing is a systematic pattern - spokes forming an "
                    "ellipse, with opposite directions agreeing and adjacent "
                    "directions differing. The fitted model above is isotropic "
                    "regardless, and kriging built from it will be too."
                )
                aniso_color = "#dd6b20"
            elif np.isfinite(rose["max_min_ratio"]):
                aniso_note = (
                    f"The fitted range varies by a factor of "
                    f"{rose['max_min_ratio']:.2f} across directions, which is "
                    "what an isotropic field looks like: no direction is "
                    "markedly more continuous than another."
                )
                aniso_color = "#2f855a"
            else:
                aniso_note = (
                    "Fewer than two directions could be fitted, so no "
                    "longest-to-shortest comparison could be made. Directions "
                    "that could not be fitted are unmeasured, not unstructured."
                )
                aniso_color = "#718096"

            if n_fitted_dirs < len(rose["angles_deg"]):
                aniso_note += (
                    f" Only {n_fitted_dirs} of "
                    f"{len(rose['angles_deg'])} directions carried enough point "
                    "pairs to fit a range; the rest are marked as unmeasured in "
                    "the table below rather than counted as having no structure."
                )
            aniso_note += rose_model_note

            rows = []
            for k, ang in enumerate(rose["angles_deg"]):
                rng_k = rose["ranges"][k]
                n_pairs = (int(directions[k][2].sum())
                           if directions and k < len(directions) else 0)
                rows.append(
                    "<tr>"
                    f'<td class="metric-name">{ang:.1f}&#176;</td>'
                    f'<td class="metric-val">{n_pairs}</td>'
                    f'<td class="metric-val">'
                    f'{"-" if not np.isfinite(rng_k) else format(rng_k, ".6g")}'
                    "</td></tr>"
                )
            dir_table = (
                "<table><thead><tr><th>Direction (math. deg)</th>"
                "<th>Pairs in &plusmn; tolerance cone</th>"
                "<th>Fitted range (map units)</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )
        else:
            rose_svg = ""
            dir_table = ""
            aniso_note = (
                "Directional variograms were skipped for this run, so "
                "anisotropy was not assessed. Re-run with the direction count "
                "above 0 to check whether the isotropic assumption holds."
            )
            aniso_color = "#718096"

        bin_rows = []
        for k in range(len(lag)):
            bin_rows.append(
                "<tr>"
                f'<td class="metric-val">{lag[k]:.6g}</td>'
                f'<td class="metric-val">{int(npairs[k])}</td>'
                f'<td class="metric-val">{semivariance[k]:.6g}</td>'
                "</tr>"
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

        if np.isfinite(n_ratio) and n_ratio > 0.8:
            next_action = (
                "Most of the variation is at a scale finer than the sampling. "
                "Kriging will interpolate smoothly but add little local detail; "
                "consider whether the sampling density can be improved before "
                "trusting the interpolated surface."
            )
        elif np.isfinite(r2) and r2 < 0.5:
            next_action = (
                "Try the other theoretical models and a different number of "
                "distance bins before kriging. A weak fit here propagates "
                "directly into the interpolated surface."
            )
        else:
            next_action = (
                "Pass these parameters to Ordinary / Universal Kriging "
                "Interpolation using the fitted-parameter layer this tool "
                "produced, then read the leave-one-out cross-validation "
                "statistics it reports before trusting the surface."
            )

        guidance_html = analyst_guidance_html(
            "Empirical Semivariogram & Model Fitting",
            "Estimates how spatial similarity decays with distance, and fits a "
            "theoretical model whose nugget, sill and range parameterise the "
            "field for kriging.",
            [
                "The input layer is in a projected CRS, so distances are in "
                "metres or feet rather than degrees.",
                "The target field has enough valid numeric values and few "
                "skipped records.",
                "The distance bins extend far enough for the curve to reach a "
                "sill rather than still rising at the last bin.",
            ],
            [
                "Reading the range as precisely identified - it trades off "
                "against the nugget and sill and can be 20-30 per cent off even "
                "when the fitted curve looks good.",
                "Choosing the maximum lag at or beyond half the maximum "
                "separation, where the surviving point pairs are too few for "
                "the binned value to mean much.",
                "Assuming the fitted model absorbs anisotropy. It does not: the "
                "rose is a diagnostic, and the model stays isotropic.",
            ],
            [
                "Ordinary / Universal Kriging Interpolation to build a surface "
                "from these parameters",
                "Global Moran's I to test for spatial autocorrelation as a "
                "statistic rather than a model",
                "Incremental Spatial Autocorrelation to choose a distance band "
                "on evidence instead of by eye",
            ],
            "Run this before kriging, not after: the cross-validation error "
            "kriging reports is conditional on the model fitted here, so a weak "
            "fit limits how much the validation can tell you.",
        )

        scope_note = (
            "maximum lag was set by hand"
            if lag_was_manual else
            "maximum lag defaulted to half the largest pairwise separation "
            "(Journel &amp; Huijbregts, 1978)"
        )
        caveat_text = [
            "The model is fitted to binned averages of the point-pair "
            "differences, so it inherits their sampling error. The number of "
            "bins and the maximum lag are analyst choices, and both move the "
            "fitted nugget, sill and range.",
            "The parameters describe the field as it was sampled at these "
            "locations. Carrying them to a different area, a different "
            "sampling density or a different support size is an assumption the "
            "fit cannot support on its own; re-estimate where the data allow.",
        ]
        nugget_share_txt = (
            "n/a" if not np.isfinite(n_ratio)
            else format(n_ratio * 100.0, ".1f") + " %"
        )
        r2_txt = "n/a" if not np.isfinite(r2) else format(r2, ".6f")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab Semivariogram Model Report</title>
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
    .aniso-box {{ background: #fffaf0; border-left: 5px solid {aniso_color}; padding: 16px 18px; border-radius: 4px; }}
    .flag-list {{ margin: 0 0 12px 0; padding-left: 22px; }}
    .flag-list li {{ margin-bottom: 8px; color: #4a5568; }}
    {analyst_guidance_css()}
    {chart_css()}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>Empirical Semivariogram &amp; Model Fitting</h1>
        <p class="subtitle">Field Modelled: <strong>{html.escape(field_name)}</strong> | Valid Features: <strong>{n}</strong> | Model: <strong>{html.escape(model)}</strong> | Distance Bins Used: <strong>{fitted['bins_used']}</strong></p>
    </header>

    <div class="interpretation-box">
        <h2 class="status-title">Fit Quality: {verdict}</h2>
        <p class="status-desc">{desc}</p>
    </div>

    <section>
        <h2>Executive Summary</h2>
        <p>The empirical semivariogram measures how quickly the resemblance between two locations decays with the distance separating them: pairs of points are grouped into distance bins, and each bin takes the average of half the squared difference in their values. A theoretical model is then fitted to those binned values. Its three parameters carry the spatial meaning: the <strong>nugget</strong> is the jump at zero distance (variation finer than the sampling, plus measurement error), the <strong>sill</strong> is the level the curve settles at (approximately the field's variance), and the <strong>range</strong> is the distance beyond which two locations are effectively unrelated. Fitting uses Cressie (1985) weights, which downweight the long-lag bins that rest on fewest point pairs. The {scope_note}, and distances are expressed in the layer's map units.</p>
    </section>

    <table>
        <thead><tr><th>Fitted Model Parameter</th><th>Value</th></tr></thead>
        <tbody>
            <tr><td class="metric-name">Theoretical model</td><td class="metric-val">{html.escape(model)}</td></tr>
            <tr><td class="metric-name">Nugget</td><td class="metric-val">{fitted['nugget']:.6g}</td></tr>
            <tr><td class="metric-name">Partial sill</td><td class="metric-val">{fitted['psill']:.6g}</td></tr>
            <tr><td class="metric-name">Sill (nugget + partial sill)</td><td class="metric-val">{fitted['sill']:.6g}</td></tr>
            <tr><td class="metric-name">Range (practical, map units)</td><td class="metric-val">{fitted['range']:.6g}</td></tr>
            <tr><td class="metric-name">Nugget as share of sill</td><td class="metric-val">{nugget_share_txt}</td></tr>
            <tr><td class="metric-name">Stable shape exponent (alpha)</td><td class="metric-val">{alpha:.6g}</td></tr>
            <tr><td class="metric-name">Goodness of fit (weighted R2)</td><td class="metric-val">{r2_txt}</td></tr>
            <tr><td class="metric-name">Root mean squared residual</td><td class="metric-val">{fitted['rmse']:.6g}</td></tr>
            <tr><td class="metric-name">Populated distance bins used</td><td class="metric-val">{fitted['bins_used']}</td></tr>
        </tbody>
    </table>

    <section>
        <h2>Empirical Semivariogram and Fitted Model</h2>
        {plot}
    </section>

    <section>
        <h2>Binned Semivariance</h2>
        <table>
            <thead><tr><th>Mean lag distance</th><th>Point pairs in bin N(h)</th><th>Mean semivariance</th></tr></thead>
            <tbody>{"".join(bin_rows)}</tbody>
        </table>
        <p>Bins resting on few pairs are the least reliable. A curve still rising at the final bin means the range has not been reached within the distances sampled, so the fitted range is constrained by the data extent rather than by the field.</p>
    </section>

    <section>
        <h2>Directional Variograms and Anisotropy</h2>
        {rose_svg}
        <div class="aniso-box"><p style="margin:0;">{aniso_note}</p></div>
        {dir_table}
        <p>Directions are given in the mathematical convention: 0&#176; points east and angles increase counter-clockwise. Each cone also contains the opposite direction, because the semivariogram is symmetric under a 180&#176; rotation.</p>
    </section>

    {flag_html}

    {diagnostics_html(numeric_summary, None, crs_warning)}

    <section>
        <h2>Recommended Next Action</h2>
        <div class="next-action">{html.escape(next_action)}</div>
    </section>

    {caveats_html("Semivariogram model fitting", None, numeric_summary, caveat_text)}

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

    @style_loaded_output
    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}

        layer = context.getMapLayer(self.out_layer_id)
        if not layer:
            return {}

        apply_output_metadata(
            layer,
            "PlanX GeoStats fitted semivariogram parameters (input to kriging)",
            {
                "model": "Fitted theoretical semivariogram model",
                "nugget": "Nugget: discontinuity at zero distance",
                "psill": "Partial sill",
                "sill": "Sill: nugget + partial sill",
                "range": "Practical range in map units",
                "alpha": "Shape exponent (stable model only)",
                "r2": "Weighted goodness of fit of the model to the binned variogram",
                "rmse": "Root mean squared residual of the fit",
                "bins": "Populated distance bins used in the fit",
                "npoints": "Input features that entered the variogram",
                "dir_ratio": "Longest-to-shortest fitted directional range (NULL when not assessed)",
            },
            self.displayName(),
        )

        return {}


def _single_point(coords_arr):
    """A single point at the mean centre of the input, as the parameter anchor.

    The parameter layer needs a geometry to travel through Processing's sink
    machinery and to be visible on the map; the mean centre is the least
    misleading choice, since the fitted model describes the whole field rather
    than any one observation.
    """
    return QgsGeometry.fromPointXY(
        QgsPointXY(float(coords_arr[:, 0].mean()), float(coords_arr[:, 1].mean()))
    )
