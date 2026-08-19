# -*- coding: utf-8 -*-
"""Multiscale Geographically Weighted Regression Processing Algorithm."""
from __future__ import annotations

import html
import logging
import math
import os
import tempfile

import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsGraduatedSymbolRenderer,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsRendererRange,
    QgsSymbol,
)

from ..core.analysis_diagnostics import (
    crs_unit_warning,
    filter_weights_to_valid_ids,
    format_number,
    push_residual_spatial_diagnostics,
    regression_quality_html,
    regression_quality_summary,
    residual_spatial_autocorrelation_html,
    residual_spatial_autocorrelation_summary,
)
from ..dependencies import optional_dependency_error
from ..core.layer_metadata import apply_output_metadata
from ..core.weights import build_weights_matrix, geometry_centroid_point
from ..core.charts import chart_css, scatter_plot_svg

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class MGWRAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    DEP_VAR = "DEP_VAR"
    INDEPENDENTS = "INDEPENDENTS"
    KERNEL_TYPE = "KERNEL_TYPE"
    CRITERION = "CRITERION"
    MIN_BW = "MIN_BW"
    MAX_BW = "MAX_BW"
    MAX_ITER = "MAX_ITER"
    N_CHUNKS = "N_CHUNKS"
    SPHERICAL = "SPHERICAL"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    KERNEL_LABELS = [
        "Adaptive bisquare",
        "Adaptive gaussian",
        "Adaptive exponential",
        "Fixed bisquare",
        "Fixed gaussian",
        "Fixed exponential",
    ]
    KERNEL_KEYS = [
        ("bisquare", False),
        ("gaussian", False),
        ("exponential", False),
        ("bisquare", True),
        ("gaussian", True),
        ("exponential", True),
    ]
    CRITERIA = ["AICc", "AIC", "BIC", "CV"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "multiscale_geographically_weighted_regression"

    def displayName(self) -> str:
        return "Multiscale Geographically Weighted Regression (MGWR)"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("multiscale_geographically_weighted_regression")

    def createInstance(self):
        return MGWRAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Fits Multiscale Geographically Weighted Regression using the "
            "PySAL mgwr package: like GWR, it estimates a local coefficient "
            "surface per variable, but unlike GWR's single shared "
            "bandwidth, MGWR lets each explanatory variable search for and "
            "use its own optimal bandwidth (via mgwr's golden-section "
            "Sel_BW search, up to the configured maximum search "
            "iterations). This matters when different relationships "
            "genuinely operate at different spatial scales in the same "
            "model - e.g. parcel-level accessibility effects that vary "
            "block by block alongside district-level socioeconomic effects "
            "that vary much more slowly across the study area.\n\n"
            "Output: predicted values, residuals, standardized residuals, "
            "the selected bandwidth range, and per-variable local "
            "coefficients, standard errors, and t-values; the HTML report "
            "summarizes the selected bandwidth for each variable "
            "individually - the headline MGWR result (a variable whose "
            "selected bandwidth approaches the full sample size is behaving "
            "essentially globally, while a small bandwidth means a "
            "genuinely local, fast-varying relationship) - plus model fit, "
            "warnings, and caveats.\n\n"
            "This is the most computationally expensive tool in the group "
            "and the bandwidth search can be unstable with small samples or "
            "highly correlated predictors - the report explicitly flags "
            "small-sample runs as exploratory. Requires the optional mgwr "
            "package (installable via Install / Update GeoStats Libraries). "
            "Compare MGWR against plain GWR and against Model Comparison "
            "Matrix before concluding that variable-specific bandwidths are "
            "actually warranted - the added complexity is only worth it "
            "when the per-variable bandwidths genuinely differ."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.DEP_VAR,
                "Dependent variable field",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.INDEPENDENTS,
                "Explanatory variable fields",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.KERNEL_TYPE,
                "Kernel and bandwidth type",
                options=self.KERNEL_LABELS,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.CRITERION,
                "Bandwidth selection criterion",
                options=self.CRITERIA,
                defaultValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_BW,
                "Minimum bandwidth or neighbor count (0 = automatic lower bound)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_BW,
                "Maximum bandwidth or neighbor count (0 = automatic upper bound)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MAX_ITER,
                "Maximum multiscale bandwidth-search iterations",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=50,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.N_CHUNKS,
                "Fit chunks for memory control",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=1,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.SPHERICAL,
                "Use spherical distance for longitude/latitude coordinates",
                defaultValue=False,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output MGWR local diagnostics layer"))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT,
                "Output MGWR HTML report",
                fileFilter="HTML files (*.html)",
                optional=True,
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", "MGWR diagnostic report"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        sel_bw_class, mgwr_class = self._load_mgwr_dependencies()
        dep_var = self.parameterAsString(parameters, self.DEP_VAR, context)
        indep_fields = self.parameterAsFields(parameters, self.INDEPENDENTS, context)
        if not indep_fields:
            raise QgsProcessingException("At least one explanatory variable must be selected.")

        kernel_idx = self.parameterAsEnum(parameters, self.KERNEL_TYPE, context)
        kernel, fixed = self.KERNEL_KEYS[kernel_idx]
        kernel_label = self.KERNEL_LABELS[kernel_idx]
        criterion = self.CRITERIA[self.parameterAsEnum(parameters, self.CRITERION, context)]
        min_bw = self.parameterAsDouble(parameters, self.MIN_BW, context)
        max_bw = self.parameterAsDouble(parameters, self.MAX_BW, context)
        max_iter = self.parameterAsInt(parameters, self.MAX_ITER, context)
        n_chunks = self.parameterAsInt(parameters, self.N_CHUNKS, context)
        spherical = self.parameterAsBoolean(parameters, self.SPHERICAL, context)

        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), "planx_mgwr_report.html")

        dep_idx = source.fields().lookupField(dep_var)
        exp_indices = [source.fields().lookupField(name) for name in indep_fields]
        if dep_idx < 0:
            raise QgsProcessingException(f"Dependent field '{dep_var}' not found.")
        missing = [name for name, idx in zip(indep_fields, exp_indices) if idx < 0]
        if missing:
            raise QgsProcessingException(f"Explanatory fields not found: {', '.join(missing)}")

        feedback.pushInfo("Extracting complete numeric records and centroid coordinates for MGWR...")
        y_values = []
        x_rows = []
        coords = []
        valid_fids = []
        skipped = 0
        total = source.featureCount() or 1
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            geometry = feature.geometry()
            if geometry is None or geometry.isEmpty():
                skipped += 1
                continue
            dep_value = self._to_float(feature.attribute(dep_idx))
            exp_values = [self._to_float(feature.attribute(field_idx)) for field_idx in exp_indices]
            if dep_value is None or any(value is None for value in exp_values):
                skipped += 1
                continue
            point = geometry_centroid_point(geometry)
            if point is None:
                skipped += 1
                continue
            y_values.append(dep_value)
            x_rows.append(exp_values)
            coords.append([point.x(), point.y()])
            valid_fids.append(int(feature.id()))
            feedback.setProgress(int(15 * (idx / total)))

        n = len(y_values)
        p = len(indep_fields)
        if n <= p + 3:
            raise QgsProcessingException(
                f"Insufficient complete records ({n}). MGWR needs more observations than "
                f"the intercept, explanatory variables, and multiscale bandwidth parameters."
            )

        y = np.array(y_values, dtype=float).reshape((-1, 1))
        x_data = np.array(x_rows, dtype=float)
        coords_array = np.array(coords, dtype=float)
        if float(np.std(y[:, 0])) <= 1e-12:
            raise QgsProcessingException("The dependent variable is constant across complete records.")

        model_quality = regression_quality_summary(y[:, 0], x_data, indep_fields, source.featureCount())
        crs_warning = crs_unit_warning(source)
        feedback.pushInfo(
            "MGWR model quality diagnostics: "
            f"{model_quality['used_records']} complete record(s), "
            f"{model_quality['skipped_records']} skipped record(s), "
            f"{model_quality['predictor_count']} predictor(s)."
        )
        if n < max(30, (p + 1) * 8):
            feedback.pushWarning(
                "The MGWR sample is small for multiscale bandwidth search. Treat bandwidths and local coefficients as exploratory."
            )
        if not fixed:
            # mgwr's multiscale backfitting seeds itself with an initial global bandwidth
            # search (mgwr.search.multi_bw -> Sel_BW._init_section) that does NOT honor
            # this tool's Min/Max bandwidth parameters and instead falls back to a fixed
            # "40 + 2*n_vars" lower bound for adaptive (non-fixed) kernels. When the sample
            # is smaller than that bound, the golden-section search receives an invalid
            # (lower > upper) bracket and mgwr fails deep inside its kernel code with an
            # opaque "kth out of bounds" numpy error. Catch that precondition here with an
            # actionable message instead of letting the cryptic library error surface -
            # this is independent of Min/Max bandwidth above, which only bounds the later
            # per-variable refinement, not this initial seed step.
            n_vars = p + 1
            mgwr_min_n = 40 + 2 * n_vars
            if n < mgwr_min_n:
                raise QgsProcessingException(
                    f"MGWR's adaptive-kernel multiscale bandwidth search needs at least "
                    f"{mgwr_min_n} complete records for {p} explanatory variable(s) plus the "
                    f"intercept (a hardcoded minimum inside the mgwr library's initial "
                    f"bandwidth-seeding step, not something Min/Max bandwidth above can "
                    f"override); this layer has {n}. Use a Fixed kernel (Fixed Gaussian or "
                    f"Fixed Bisquare), select a layer with more complete records, or use "
                    f"plain GWR instead of MGWR on small samples."
                )
        for risk in model_quality["risks"]:
            feedback.pushWarning(risk)
        if crs_warning and not spherical:
            feedback.pushWarning(crs_warning)

        search_kwargs = self._build_search_kwargs(n, p, fixed, min_bw, max_bw, criterion, max_iter)
        feedback.pushInfo(
            "Selecting MGWR variable-specific bandwidths "
            f"with {criterion}, {kernel_label}, max iterations={max_iter}..."
        )
        try:
            selector = sel_bw_class(
                coords_array,
                y,
                x_data,
                multi=True,
                kernel=kernel,
                fixed=fixed,
                constant=True,
                spherical=spherical,
            )
            selector.search(**search_kwargs)
        except Exception as exc:
            raise QgsProcessingException(
                "MGWR bandwidth search failed. Review sample size, coordinate system, "
                "minimum/maximum bandwidth settings, missing values, and predictor collinearity. "
                f"Underlying error: {exc}"
            )

        feedback.pushInfo("Fitting MGWR local model. This can take several minutes on large layers...")
        try:
            model = mgwr_class(
                coords_array,
                y,
                x_data,
                selector,
                kernel=kernel,
                fixed=fixed,
                constant=True,
                spherical=spherical,
                hat_matrix=False,
            )
            results = model.fit(n_chunks=n_chunks)
        except Exception as exc:
            raise QgsProcessingException(
                "MGWR model fitting failed after bandwidth selection. "
                "Try a smaller variable set, broader bandwidth bounds, or review collinearity. "
                f"Underlying error: {exc}"
            )

        extracted = self._extract_results(results, selector, y[:, 0], indep_fields)
        residual_k = min(max(8, p + 2), n - 1)
        residual_spatial = self._residual_spatial_diagnostic(
            source,
            valid_fids,
            extracted["residuals"],
            residual_k,
            feedback,
        )
        short_names = self._short_field_names(indep_fields)
        min_selected_bw = float(np.nanmin(extracted["bandwidths"])) if len(extracted["bandwidths"]) else None
        max_selected_bw = float(np.nanmax(extracted["bandwidths"])) if len(extracted["bandwidths"]) else None

        feedback.pushInfo(
            "MGWR fitted: "
            f"R2={format_number(extracted['r2'], 6)}, "
            f"AICc={format_number(extracted['aicc'], 4)}, "
            f"bandwidth range={format_number(min_selected_bw, 3)} to {format_number(max_selected_bw, 3)}."
        )

        out_fields = source.fields()
        out_fields.append(QgsField("mgwr_pred", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("mgwr_resid", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("mgwr_std", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("mgwr_minbw", QVariant.Double, len=12, prec=3))
        out_fields.append(QgsField("mgwr_maxbw", QVariant.Double, len=12, prec=3))
        out_fields.append(QgsField("mgwr_used", QVariant.Int))
        out_fields.append(QgsField("mg_b0", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("mg_se0", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("mg_t0", QVariant.Double, len=10, prec=4))
        for short_name in short_names:
            out_fields.append(QgsField(f"mg_{short_name}", QVariant.Double, len=12, prec=6))
            out_fields.append(QgsField(f"mgs_{short_name}", QVariant.Double, len=12, prec=6))
            out_fields.append(QgsField(f"mgt_{short_name}", QVariant.Double, len=10, prec=4))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs(),
        )
        self.out_layer_id = dest_id

        result_map = {fid: row_idx for row_idx, fid in enumerate(valid_fids)}
        feedback.pushInfo("Writing MGWR local coefficients and diagnostics...")
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = int(feature.id())
            row_idx = result_map.get(fid)
            if row_idx is None:
                self._write_null_output(out_feature, short_names)
            else:
                out_feature.setAttribute("mgwr_pred", self._safe_output(extracted["predicted"][row_idx]))
                out_feature.setAttribute("mgwr_resid", self._safe_output(extracted["residuals"][row_idx]))
                out_feature.setAttribute("mgwr_std", self._safe_output(extracted["std_residuals"][row_idx]))
                out_feature.setAttribute("mgwr_minbw", self._safe_output(min_selected_bw))
                out_feature.setAttribute("mgwr_maxbw", self._safe_output(max_selected_bw))
                out_feature.setAttribute("mgwr_used", 1)
                out_feature.setAttribute("mg_b0", self._safe_output(extracted["params"][row_idx, 0]))
                out_feature.setAttribute("mg_se0", self._safe_output(extracted["standard_errors"][row_idx, 0]))
                out_feature.setAttribute("mg_t0", self._safe_output(extracted["t_values"][row_idx, 0]))
                for var_idx, short_name in enumerate(short_names):
                    col = var_idx + 1
                    out_feature.setAttribute(f"mg_{short_name}", self._safe_output(extracted["params"][row_idx, col]))
                    out_feature.setAttribute(f"mgs_{short_name}", self._safe_output(extracted["standard_errors"][row_idx, col]))
                    out_feature.setAttribute(f"mgt_{short_name}", self._safe_output(extracted["t_values"][row_idx, col]))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(55 + 45 * (current / total)))

        self._write_html(
            html_path,
            dep_var,
            indep_fields,
            kernel_label,
            criterion,
            fixed,
            spherical,
            extracted,
            model_quality,
            crs_warning,
            skipped,
            min_bw,
            max_bw,
            max_iter,
            residual_spatial,
        )
        return {self.OUTPUT: dest_id, self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

    def _load_mgwr_dependencies(self):
        try:
            from mgwr.gwr import MGWR
            from mgwr.sel_bw import Sel_BW
        except Exception as exc:
            raise QgsProcessingException(optional_dependency_error(
                "Multiscale Geographically Weighted Regression",
                ["mgwr"],
                exc,
            ))
        return Sel_BW, MGWR

    def _to_float(self, value):
        if value is None or value == NULL or str(value) == "NULL":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _build_search_kwargs(self, n, p, fixed, min_bw, max_bw, criterion, max_iter):
        kwargs = {
            "criterion": criterion,
            "max_iter_multi": int(max_iter),
            "verbose": False,
        }
        if fixed:
            if min_bw > 0:
                kwargs["multi_bw_min"] = [float(min_bw)]
            if max_bw > 0:
                if min_bw > 0 and max_bw <= min_bw:
                    raise QgsProcessingException("Maximum fixed bandwidth must be greater than minimum fixed bandwidth.")
                kwargs["multi_bw_max"] = [float(max_bw)]
            return kwargs

        lower = int(round(min_bw)) if min_bw > 0 else max(2, p + 2)
        upper = int(round(max_bw)) if max_bw > 0 else max(lower + 1, n - 1)
        if upper > n - 1:
            upper = n - 1
        if lower >= upper:
            raise QgsProcessingException(
                f"Adaptive MGWR bandwidth bounds are invalid after adjustment: minimum={lower}, maximum={upper}."
            )
        kwargs["multi_bw_min"] = [lower]
        kwargs["multi_bw_max"] = [upper]
        return kwargs

    def _extract_results(self, results, selector, y, indep_fields):
        params = self._array2d(self._safe_getattr(results, "params"))
        if params is None:
            raise QgsProcessingException("MGWR returned no local coefficient matrix.")
        n = len(y)
        predicted = self._array1d(self._safe_getattr(results, "predy"), n)
        if predicted is None:
            predicted = np.full(n, np.nan)
        residuals = self._array1d(self._safe_getattr(results, "resid_response"), n)
        if residuals is None or not np.all(np.isfinite(residuals)):
            residuals = y - predicted
        std_residuals = self._array1d(self._safe_getattr(results, "std_res"), n)
        if std_residuals is None or not np.all(np.isfinite(std_residuals)):
            resid_std = float(np.nanstd(residuals))
            std_residuals = residuals / resid_std if resid_std > 0 else np.zeros(n)

        standard_errors = self._array2d(self._safe_getattr(results, "bse"))
        if standard_errors is None or standard_errors.shape != params.shape:
            standard_errors = np.full(params.shape, np.nan)
        t_values = self._array2d(self._safe_getattr(results, "tvalues"))
        if t_values is None or t_values.shape != params.shape:
            t_values = np.divide(params, standard_errors, out=np.full(params.shape, np.nan), where=standard_errors > 0)

        bandwidths = self._extract_bandwidths(results, selector, params.shape[1])
        enp_j = self._array1d(self._safe_getattr(results, "ENP_j"), params.shape[1])
        if enp_j is None:
            enp_j = np.full(params.shape[1], np.nan)

        return {
            "params": params,
            "standard_errors": standard_errors,
            "t_values": t_values,
            "predicted": predicted,
            "residuals": residuals,
            "std_residuals": std_residuals,
            "bandwidths": bandwidths,
            "enp_j": enp_j,
            "r2": self._safe_float(self._safe_getattr(results, "R2")),
            "adj_r2": self._safe_float(self._safe_getattr(results, "adj_R2")),
            "aic": self._safe_float(self._safe_getattr(results, "aic")),
            "aicc": self._safe_float(self._safe_getattr(results, "aicc")),
            "bic": self._safe_float(self._safe_getattr(results, "bic")),
            "sigma2": self._safe_float(self._safe_getattr(results, "sigma2")),
            "tr_s": self._safe_float(self._safe_getattr(results, "tr_S")),
            "names": ["Intercept"] + list(indep_fields),
        }

    def _residual_spatial_diagnostic(self, source, valid_fids, residuals, k_neighbors, feedback):
        try:
            feedback.pushInfo(
                f"Building KNN weights with K={k_neighbors} for MGWR residual spatial autocorrelation diagnostics..."
            )
            neighbors, _, _, _ = build_weights_matrix(source, "knn", k_neighbors=k_neighbors, feedback=feedback)
            filtered_neighbors, filtered_weights, filtered_ids = filter_weights_to_valid_ids(neighbors, valid_fids)
            summary = residual_spatial_autocorrelation_summary(residuals, filtered_neighbors, filtered_weights, filtered_ids)
            push_residual_spatial_diagnostics(feedback, summary)
            return summary
        except Exception as exc:
            summary = {
                "available": False,
                "moran_i": None,
                "expected_i": None,
                "variance": None,
                "z_score": None,
                "p_value": None,
                "neighbor_summary": None,
                "status": "Not available",
                "message": str(exc),
            }
            push_residual_spatial_diagnostics(feedback, summary)
            return summary

    def _extract_bandwidths(self, results, selector, expected_size):
        candidates = [
            getattr(results, "bws", None),
            getattr(selector, "bws", None),
            getattr(selector, "bw", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            arr = None
            try:
                if isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[0], (list, tuple, np.ndarray)):
                    arr = np.asarray(candidate[0], dtype=float).flatten()
                else:
                    arr = np.asarray(candidate, dtype=float).flatten()
            except (TypeError, ValueError):
                arr = None
            if arr is None:
                continue
            if arr.size >= expected_size:
                return arr[:expected_size]
        return np.full(expected_size, np.nan)

    def _safe_getattr(self, obj, name, default=None):
        """Like getattr(), but also swallows exceptions raised while evaluating a
        property - not just a genuinely missing attribute. Several MGWRResults
        properties (std_res, bse, tvalues, ...) divide by the hat-matrix influence
        diagonal, which is None whenever hat_matrix=False (used here to avoid the
        O(n^2) hat matrix on large layers); accessing those properties then raises
        instead of returning None the way a missing attribute would, and a plain
        getattr(obj, name, default) does not catch that - only AttributeError from
        a genuinely absent attribute triggers its default."""
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    def _array1d(self, value, expected_size):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size != expected_size:
            return None
        return arr

    def _array2d(self, value):
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2:
            return None
        return arr

    def _safe_float(self, value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return result

    def _safe_output(self, value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(result):
            return None
        return result

    def _short_field_names(self, names):
        used = set()
        result = []
        for idx, name in enumerate(names, start=1):
            base = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)[:5].strip("_")
            if not base:
                base = f"v{idx}"
            candidate = base
            suffix = 1
            while candidate in used:
                suffix_text = str(suffix)
                candidate = (base[: max(1, 5 - len(suffix_text))] + suffix_text)[:5]
                suffix += 1
            used.add(candidate)
            result.append(candidate)
        return result

    def _write_null_output(self, feature, short_names):
        for name in ["mgwr_pred", "mgwr_resid", "mgwr_std", "mgwr_minbw", "mgwr_maxbw"]:
            feature.setAttribute(name, None)
        feature.setAttribute("mgwr_used", 0)
        for name in ["mg_b0", "mg_se0", "mg_t0"]:
            feature.setAttribute(name, None)
        for short_name in short_names:
            feature.setAttribute(f"mg_{short_name}", None)
            feature.setAttribute(f"mgs_{short_name}", None)
            feature.setAttribute(f"mgt_{short_name}", None)

    def _write_html(
        self,
        path,
        dep_var,
        indep_fields,
        kernel_label,
        criterion,
        fixed,
        spherical,
        results,
        model_quality,
        crs_warning,
        skipped,
        min_bw,
        max_bw,
        max_iter,
        residual_spatial,
    ):
        names = results["names"]
        bandwidth_rows = []
        for idx, name in enumerate(names):
            params = results["params"][:, idx]
            t_values = results["t_values"][:, idx]
            bandwidth_rows.append(
                "<tr>"
                f"<td><strong>{html.escape(name)}</strong></td>"
                f"<td>{format_number(results['bandwidths'][idx], 3)}</td>"
                f"<td>{format_number(results['enp_j'][idx], 4)}</td>"
                f"<td>{format_number(np.nanmin(params), 6)}</td>"
                f"<td>{format_number(np.nanmedian(params), 6)}</td>"
                f"<td>{format_number(np.nanmax(params), 6)}</td>"
                f"<td>{format_number(np.nanmedian(np.abs(t_values)), 4)}</td>"
                "</tr>"
            )

        residual_chart = scatter_plot_svg(
            np.asarray(results["predicted"], dtype=float).tolist(),
            np.asarray(results["residuals"], dtype=float).tolist(),
            x_label="Predicted value",
            y_label="Residual",
            trend_line=True,
            split_y=0.0,
        )

        crs_block = ""
        if crs_warning and not spherical:
            crs_block = f"<div class=\"note\"><strong>CRS warning:</strong> {html.escape(crs_warning)}</div>"

        bandwidth_unit = "map units" if fixed else "neighbors"
        bounds_text = "automatic"
        if min_bw > 0 or max_bw > 0:
            bounds_text = f"minimum={min_bw if min_bw > 0 else 'automatic'}, maximum={max_bw if max_bw > 0 else 'automatic'}"

        local_risk = ""
        if residual_spatial.get("available") and residual_spatial.get("p_value") is not None and residual_spatial["p_value"] < 0.05:
            local_risk = "Residuals retain a spatial pattern even after MGWR; review omitted variables, boundaries, and bandwidth settings."
        elif model_quality["risks"]:
            local_risk = "Resolve model-quality warnings before treating local coefficient patterns as stable."
        elif len(indep_fields) > 4:
            local_risk = "Review whether the variable set is too broad; MGWR bandwidths are easier to defend with a focused model."
        else:
            local_risk = "Map residuals and the strongest local coefficients, then compare their geography with planning theory and data collection context."

        content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab MGWR Diagnostics</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #25313f; background: #f6f8fb; margin: 0; padding: 24px; line-height: 1.55; }}
.container {{ max-width: 1120px; margin: 0 auto; background: #fff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 1.72rem; color: #17212f; }}
h2 {{ color: #1a202c; font-size: 1.15rem; margin: 28px 0 12px; border-left: 4px solid #2b6cb0; padding-left: 10px; }}
.subtitle {{ color: #607086; margin: 0 0 24px; }}
.summary {{ background: #eef7f3; border-left: 5px solid #2f855a; padding: 16px 18px; margin: 20px 0; }}
.note {{ background: #fff8e6; border-left: 5px solid #b7791f; padding: 14px 18px; margin: 20px 0; }}
.next-action {{ background: #f0fff4; border-left: 5px solid #2f855a; padding: 16px 18px; border-radius: 4px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0 24px; }}
.card {{ border: 1px solid #d9e2ec; background: #f8fafc; border-radius: 6px; padding: 14px; }}
.card-title {{ color: #607086; font-size: .74rem; text-transform: uppercase; font-weight: 700; }}
.card-value {{ color: #1f5f8b; font-size: 1.28rem; font-weight: 800; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 18px; }}
th, td {{ border-bottom: 1px solid #edf2f7; padding: 10px; text-align: left; vertical-align: top; font-size: .87rem; }}
th {{ background: #ebf4ff; color: #24527a; text-transform: uppercase; font-size: .72rem; letter-spacing: .05em; }}
.metric-name {{ font-weight: 700; color: #314155; }}
footer {{ margin-top: 36px; padding-top: 14px; border-top: 1px solid #edf2f7; color: #7a899c; font-size: .82rem; }}
{chart_css()}
</style>
</head>
<body>
<div class="container">
<h1>Multiscale Geographically Weighted Regression (MGWR)</h1>
<p class="subtitle">Dependent field: <strong>{html.escape(dep_var)}</strong> | Explanatory variables: <strong>{html.escape(', '.join(indep_fields))}</strong> | Kernel: <strong>{html.escape(kernel_label)}</strong></p>

<section class="summary">
<strong>Executive summary.</strong> MGWR estimates a local regression surface for each selected relationship while allowing every explanatory variable to use its own spatial scale. Smaller bandwidths indicate localized relationships; broader bandwidths indicate relationships that behave more globally across the study area. Use these bandwidths as analytical evidence about scale, not as proof of causality.
</section>
{crs_block}

<div class="grid">
<div class="card"><div class="card-title">Complete Records</div><div class="card-value">{model_quality['used_records']}</div></div>
<div class="card"><div class="card-title">R2</div><div class="card-value">{format_number(results['r2'], 6)}</div></div>
<div class="card"><div class="card-title">Adjusted R2</div><div class="card-value">{format_number(results['adj_r2'], 6)}</div></div>
<div class="card"><div class="card-title">AICc</div><div class="card-value">{format_number(results['aicc'], 4)}</div></div>
</div>

{regression_quality_html(model_quality)}
{residual_spatial_autocorrelation_html(residual_spatial)}

<h2>Residual vs. Predicted</h2>
{residual_chart}

<h2>Bandwidth Selection</h2>
<table>
<tbody>
<tr><td class="metric-name">Criterion</td><td>{html.escape(criterion)}</td></tr>
<tr><td class="metric-name">Bandwidth type</td><td>{'Fixed distance' if fixed else 'Adaptive nearest-neighbor count'}</td></tr>
<tr><td class="metric-name">Bandwidth unit</td><td>{html.escape(bandwidth_unit)}</td></tr>
<tr><td class="metric-name">Search bounds</td><td>{html.escape(bounds_text)}</td></tr>
<tr><td class="metric-name">Maximum multiscale iterations</td><td>{max_iter}</td></tr>
<tr><td class="metric-name">Spherical distance</td><td>{'Yes' if spherical else 'No'}</td></tr>
<tr><td class="metric-name">Skipped/incomplete records</td><td>{skipped}</td></tr>
</tbody>
</table>

<h2>Variable Scale and Local Coefficients</h2>
<table>
<thead><tr><th>Term</th><th>Selected Bandwidth</th><th>Effective Params</th><th>Min Coef</th><th>Median Coef</th><th>Max Coef</th><th>Median |t|</th></tr></thead>
<tbody>{''.join(bandwidth_rows)}</tbody>
</table>

<h2>Model Fit Statistics</h2>
<table>
<tbody>
<tr><td class="metric-name">AIC</td><td>{format_number(results['aic'], 6)}</td></tr>
<tr><td class="metric-name">AICc</td><td>{format_number(results['aicc'], 6)}</td></tr>
<tr><td class="metric-name">BIC</td><td>{format_number(results['bic'], 6)}</td></tr>
<tr><td class="metric-name">Sigma squared</td><td>{format_number(results['sigma2'], 6)}</td></tr>
<tr><td class="metric-name">Trace S</td><td>{format_number(results['tr_s'], 6)}</td></tr>
</tbody>
</table>

<h2>Interpretation</h2>
<p>MGWR is most useful when different explanatory variables plausibly operate at different planning scales. A local accessibility measure may have a small bandwidth, while a socioeconomic or market variable may have a much broader bandwidth. Coefficient maps show where the relationship is stronger, weaker, positive, or negative; t-value maps help screen where the local estimate is more stable.</p>

<h2>Recommended Analyst Action</h2>
<div class="next-action">{html.escape(local_risk)} Compare MGWR against OLS, GWR, and residual diagnostics before making a planning recommendation.</div>

<h2>Caveats</h2>
<ul>
<li>MGWR is computationally expensive and exploratory; bandwidth search can be unstable with small samples or highly correlated predictors.</li>
<li>Selected bandwidths depend on the coordinate system, kernel, criterion, and bounds. Document these settings when reporting results.</li>
<li>Local coefficient variation is not automatically causal evidence. Use policy context, data lineage, and residual maps to decide whether the pattern is credible.</li>
<li>Fixed-distance MGWR should use a projected CRS with meaningful map units unless spherical distance is intentionally enabled for longitude/latitude coordinates.</li>
</ul>

<footer>Generated by PlanX GeoStats Lab local spatial statistics engine.</footer>
</div>
</body>
</html>"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}
        layer = context.getMapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying MGWR standardized residual styling...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats MGWR multiscale local model output",
            {
                "mgwr_pred": "MGWR predicted dependent-variable value",
                "mgwr_resid": "Observed minus predicted MGWR residual",
                "mgwr_std": "Standardized MGWR residual used for diverging residual maps",
                "mgwr_minbw": "Minimum selected MGWR bandwidth across model terms",
                "mgwr_maxbw": "Maximum selected MGWR bandwidth across model terms",
                "mgwr_used": "1 when the feature was used in the fitted model, otherwise 0",
                "mg_b0": "MGWR intercept coefficient",
                "mg_se0": "MGWR intercept standard error",
                "mg_t0": "MGWR intercept t statistic",
            },
            self.displayName(),
        )
        ranges = []
        range_definitions = [
            (-9999.0, -2.5, "#2166ac", "< -2.5 Std Residual"),
            (-2.5, -1.5, "#67a9cf", "-2.5 to -1.5 Std Residual"),
            (-1.5, -0.5, "#d1e5f0", "-1.5 to -0.5 Std Residual"),
            (-0.5, 0.5, "#f7f7f7", "-0.5 to 0.5 Std Residual"),
            (0.5, 1.5, "#fddbc7", "0.5 to 1.5 Std Residual"),
            (1.5, 2.5, "#f4a582", "1.5 to 2.5 Std Residual"),
            (2.5, 9999.0, "#b2182b", "> 2.5 Std Residual"),
        ]
        for min_value, max_value, color_hex, label in range_definitions:
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)
            if symbol.symbolLayerCount() > 0:
                symbol_layer = symbol.symbolLayer(0)
                if hasattr(symbol_layer, "setStrokeColor"):
                    symbol_layer.setStrokeColor(QColor("#b0b0b0"))
                if hasattr(symbol_layer, "setStrokeWidth"):
                    symbol_layer.setStrokeWidth(0.1)
                if hasattr(symbol_layer, "setOutlineColor"):
                    symbol_layer.setOutlineColor(QColor("#b0b0b0"))
            ranges.append(QgsRendererRange(min_value, max_value, symbol, label))
        layer.setRenderer(QgsGraduatedSymbolRenderer("mgwr_std", ranges))
        layer.triggerRepaint()
        return {}
