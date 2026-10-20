# -*- coding: utf-8 -*-
"""Shared scaffolding for the v3.11+ spatial-statistics tools.

The tools added from v3.11.0 on share one shape: read a vector layer and some
numeric fields, build a weights graph, run a numpy engine from ``core``, copy
the input features to an output layer with the result fields appended, write
an HTML report and style the output in ``postProcessAlgorithm``.  The older
tools each repeat that boilerplate in full; these keep only what differs -
their parameters, their computation and their report - and inherit the rest,
so a fix to the shared part lands in all of them at once.

A subclass sets the class attributes, adds its own parameters in
``add_parameters`` and implements ``compute``, which returns a
:class:`ToolResult`.
"""
from __future__ import annotations

import html
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    NULL,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputHtml,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)

from ..core.analysis_diagnostics import crs_unit_warning
from ..core.charts import chart_css
from ..core.layer_metadata import apply_output_metadata
from ..core.local_stats_engines import SpatialGraph
from ..core.reporting import analyst_guidance_css
from ..core.symbology import (
    apply_renderer,
    categorized_renderer,
    categorical_field_renderer,
    categorical_id_renderer,
    diverging_residual_renderer,
    lisa_quadrant_renderer,
    sequential_quantile_renderer,
)
from ._deliver import style_loaded_output
from ._icons import algorithm_icon
from ._mixins import HelpUrlMixin

WEIGHT_OPTIONS = ["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"]
WEIGHT_CODES = ["queen", "rook", "knn", "distance"]

DOUBLE = QVariant.Double
INT = QVariant.Int
STRING = QVariant.String


@dataclass
class ToolResult:
    """What ``compute`` hands back to the shared writer."""

    report_html: str
    fields: list = field(default_factory=list)        # [(name, QVariant type)]
    rows: dict = field(default_factory=dict)          # fid -> {name: value}
    field_docs: dict = field(default_factory=dict)    # name -> description
    style: tuple | None = None                        # renderer spec, see _renderer_for
    layer_title: str = ""
    outputs: dict = field(default_factory=dict)       # extra Processing outputs


def html_table(headers, rows, numeric_precision: int = 4) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "&mdash;"
            return f"{float(value):.{numeric_precision}g}"
        return html.escape(str(value))

    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f'<table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def report_page(title: str, subtitle: str, body: str) -> str:
    """The page shell every v3.11+ report uses (same look as the LISA reports)."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PlanX GeoStats Lab - {html.escape(title)}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #2d3748; background-color: #f7fafc; margin: 0; padding: 20px; line-height: 1.5; }}
    .container {{ max-width: 860px; margin: 0 auto; background: #ffffff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); }}
    header {{ border-bottom: 2px solid #edf2f7; padding-bottom: 20px; margin-bottom: 25px; }}
    h1 {{ color: #1a202c; margin: 0 0 5px 0; font-size: 1.6rem; }}
    .subtitle {{ color: #718096; margin: 0; font-size: 0.95rem; }}
    section {{ margin: 28px 0; }}
    h2 {{ color: #1a202c; font-size: 1.15rem; margin: 0 0 12px 0; }}
    p {{ margin: 8px 0; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; margin: 10px 0; }}
    .data-table th, .data-table td {{ border: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; }}
    .data-table th {{ background: #f1f5f9; }}
    .callout {{ background: #fffbeb; border-left: 4px solid #d97706; padding: 10px 14px; border-radius: 4px; margin: 12px 0; }}
    .callout.info {{ background: #eff6ff; border-left-color: #2563eb; }}
    .note {{ margin-top: 12px; }}
    footer {{ margin-top: 40px; border-top: 1px solid #edf2f7; padding-top: 15px; font-size: 0.8rem; color: #a0aec0; text-align: center; }}
    {analyst_guidance_css()}
    {chart_css()}
</style>
</head>
<body>
<div class="container">
<header>
<h1>{html.escape(title)}</h1>
<p class="subtitle">{subtitle}</p>
</header>
{body}
<footer>Generated by PlanX GeoStats Lab spatial statistics engine.</footer>
</div>
</body>
</html>
"""


def with_notes(report_html: str, notes) -> str:
    """Insert warning callouts right under a report_page header."""
    notes = [n for n in notes if n]
    if not notes:
        return report_html
    block = "".join(callout(html.escape(n)) for n in notes)
    return report_html.replace("</header>", "</header>" + block, 1)


def callout(text: str, info: bool = False) -> str:
    return f'<div class="callout{" info" if info else ""}">{text}</div>'


def fmt(value, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"


class SpatialStatBase(HelpUrlMixin, QgsProcessingAlgorithm):
    """Base class: input layer, optional weights, output layer and report."""

    INPUT = "INPUT"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    OUTPUT = "OUTPUT"
    HTML_REPORT = "HTML_REPORT"

    ALG_ID = ""
    DISPLAY = ""
    GROUP = ""
    GROUP_ID = ""
    HELP = ""
    USES_WEIGHTS = True
    DEFAULT_WEIGHT = 0
    DEFAULT_KNN = 8
    HAS_LAYER_OUTPUT = True
    OUTPUT_LABEL = "Output layer"
    REPORT_LABEL = "HTML report"
    INPUT_TYPES = (QgsProcessing.TypeVectorAnyGeometry,)

    def __init__(self):
        super().__init__()
        self.out_layer_id = None
        self._result: ToolResult | None = None

    # -- identity ----------------------------------------------------------
    def name(self) -> str:
        return self.ALG_ID

    def displayName(self) -> str:
        return self.DISPLAY

    def group(self) -> str:
        return self.GROUP

    def groupId(self) -> str:
        return self.GROUP_ID

    def icon(self):
        return algorithm_icon(self.ALG_ID)

    def createInstance(self):
        return type(self)()

    def shortHelpString(self) -> str:
        return self.HELP

    # -- parameters --------------------------------------------------------
    def numeric_field(self, key, label, optional=False, multiple=False):
        self.addParameter(
            QgsProcessingParameterField(
                key, label, parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=multiple, optional=optional,
            )
        )

    def any_field(self, key, label, optional=False):
        self.addParameter(
            QgsProcessingParameterField(
                key, label, parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Any, optional=optional,
            )
        )

    def integer(self, key, label, default, minimum=None, maximum=None, optional=False):
        kwargs = {"type": QgsProcessingParameterNumber.Integer, "defaultValue": default, "optional": optional}
        if minimum is not None:
            kwargs["minValue"] = minimum
        if maximum is not None:
            kwargs["maxValue"] = maximum
        self.addParameter(QgsProcessingParameterNumber(key, label, **kwargs))

    def double(self, key, label, default, minimum=None, maximum=None, optional=False):
        kwargs = {"type": QgsProcessingParameterNumber.Double, "defaultValue": default, "optional": optional}
        if minimum is not None:
            kwargs["minValue"] = minimum
        if maximum is not None:
            kwargs["maxValue"] = maximum
        self.addParameter(QgsProcessingParameterNumber(key, label, **kwargs))

    def enum(self, key, label, options, default=0, multiple=False):
        self.addParameter(
            QgsProcessingParameterEnum(key, label, options=list(options), defaultValue=default, allowMultiple=multiple)
        )

    def add_parameters(self):  # pragma: no cover - overridden
        """Subclasses add their own parameters here (between input and weights)."""

    def add_trailing_parameters(self):
        """Parameters that belong after the weights block (permutations, seed...)."""

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", list(self.INPUT_TYPES))
        )
        self.add_parameters()
        if self.USES_WEIGHTS:
            self.enum(self.WEIGHT_TYPE, "Spatial relationship / weights type", WEIGHT_OPTIONS, self.DEFAULT_WEIGHT)
            self.integer(self.KNN, "Number of neighbors (K value, KNN only)", self.DEFAULT_KNN, minimum=1)
            self.double(self.DISTANCE_BAND, "Distance band threshold (map units, Distance Band only)", 1000.0, minimum=0.0001)
        self.add_trailing_parameters()
        if self.HAS_LAYER_OUTPUT:
            self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, self.OUTPUT_LABEL))
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.HTML_REPORT, "Output HTML report", fileFilter="HTML files (*.html)", optional=True
            )
        )
        self.addOutput(QgsProcessingOutputHtml("HTML_REPORT_OUT", self.REPORT_LABEL))

    def field_list(self, parameters, key, context):
        """Selected field names - ``parameterAsStrings`` where QGIS has it (3.32+),
        ``parameterAsFields`` on older releases, which deprecate it later."""
        if hasattr(self, "parameterAsStrings"):
            return list(self.parameterAsStrings(parameters, key, context) or [])
        return list(self.parameterAsFields(parameters, key, context) or [])

    # -- reading -----------------------------------------------------------
    @staticmethod
    def read_matrix(source, field_names, feedback, allow_text=False):
        """Return ``(fids, matrix, dropped)`` for features with every field usable.

        A feature with a null, non-numeric or non-finite value in any listed
        field is dropped - never read as zero - and counted, so the report's
        observation count is the number of rows that entered the computation.
        """
        indices = []
        for name in field_names:
            idx = source.fields().lookupField(name)
            if idx < 0:
                raise QgsProcessingException(f"Field '{name}' not found.")
            indices.append(idx)
        fids, rows, dropped = [], [], 0
        for feature in source.getFeatures():
            if feedback is not None and feedback.isCanceled():
                break
            row = []
            ok = True
            for name in field_names:
                raw = feature.attribute(name)
                if raw is None or raw == NULL or str(raw) == "NULL":
                    ok = False
                    break
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not np.isfinite(value):
                    ok = False
                    break
                row.append(value)
            if not ok:
                dropped += 1
                continue
            fids.append(feature.id())
            rows.append(row)
        matrix = np.asarray(rows, dtype=float).reshape(len(rows), len(field_names))
        return fids, matrix, dropped

    def read_weights(self, source, parameters, context, feedback, fids):
        from ..core.weights import build_weights_matrix

        code = WEIGHT_CODES[self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)]
        k = self.parameterAsInt(parameters, self.KNN, context)
        band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        feedback.pushInfo("Building the spatial weights graph...")
        neighbors, weights, _ids, _w = build_weights_matrix(
            source, code, k_neighbors=k, distance_band=band, feedback=feedback
        )
        graph = SpatialGraph.from_dicts(neighbors, weights, fids)
        label = WEIGHT_OPTIONS[WEIGHT_CODES.index(code)]
        if code == "knn":
            label += f" (k = {k})"
        elif code == "distance":
            label += f" ({band:g} map units)"
        islands = int(len(graph.islands))
        if islands:
            feedback.pushWarning(
                f"{islands} feature(s) have no neighbour among the usable features; their local "
                "statistic is undefined and is written as NULL. Widen the neighbourhood if this is unexpected."
            )
        if graph.n and graph.cardinality.max() == 0:
            raise QgsProcessingException(
                "No feature has a neighbour under this weights definition. Queen and Rook need "
                "polygons that touch; for points use K-Nearest Neighbors or a Distance Band."
            )
        warning = crs_unit_warning(source)
        if warning:
            feedback.pushWarning(warning)
        return graph, label

    @staticmethod
    def require(condition, message):
        if not condition:
            raise QgsProcessingException(message)

    # -- run -----------------------------------------------------------------
    def compute(self, source, parameters, context, feedback) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")
        html_path = self.parameterAsFileOutput(parameters, self.HTML_REPORT, context)
        if not html_path:
            html_path = os.path.join(tempfile.gettempdir(), f"planx_{self.ALG_ID}_report.html")

        result = self.compute(source, parameters, context, feedback)
        if feedback.isCanceled():
            return {}
        self._result = result
        outputs = {self.HTML_REPORT: html_path, "HTML_REPORT_OUT": html_path}

        if self.HAS_LAYER_OUTPUT:
            out_fields = source.fields()
            existing = {f.name().lower() for f in out_fields}
            for name, kind in result.fields:
                if name.lower() in existing:
                    raise QgsProcessingException(
                        f"The input layer already has a field named '{name}'. Rename it "
                        "(or run on a copy without that field) so the result is not mixed with old values."
                    )
                if kind == STRING:
                    out_fields.append(QgsField(name, kind, len=40))
                else:
                    out_fields.append(QgsField(name, kind))
            sink, dest_id = self.parameterAsSink(
                parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs()
            )
            if sink is None:
                raise QgsProcessingException("Could not create the output layer.")
            self.out_layer_id = dest_id
            names = [name for name, _ in result.fields]
            total = max(source.featureCount(), 1)
            for current, feature in enumerate(source.getFeatures()):
                if feedback.isCanceled():
                    break
                out = QgsFeature(feature)
                out.setFields(out_fields)
                values = result.rows.get(feature.id(), {})
                for name in names:
                    value = values.get(name)
                    if isinstance(value, (float, np.floating)):
                        value = float(value) if np.isfinite(value) else None
                    elif isinstance(value, (np.integer,)):
                        value = int(value)
                    elif isinstance(value, (np.str_,)):
                        value = str(value)
                    out.setAttribute(name, value)
                sink.addFeature(out, QgsFeatureSink.FastInsert)
                feedback.setProgress(int(80 + 20 * current / total))
            outputs[self.OUTPUT] = dest_id

        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(result.report_html)
        outputs.update(result.outputs)
        return outputs

    # -- styling -------------------------------------------------------------
    @staticmethod
    def _renderer_for(layer, spec):
        if not spec:
            return None
        kind, field_name = spec[0], spec[1]
        geometry = layer.geometryType()
        if kind == "lisa":
            return lisa_quadrant_renderer(geometry, field_name)
        if kind == "categorized":
            return categorized_renderer(geometry, field_name, spec[2])
        if kind == "quantile":
            return sequential_quantile_renderer(layer, geometry, field_name, *spec[2:])
        if kind == "diverging":
            return diverging_residual_renderer(layer, geometry, field_name)
        if kind == "categorical":
            return categorical_field_renderer(layer, geometry, field_name)
        if kind == "cluster_id":
            return categorical_id_renderer(layer, geometry, field_name)
        return None

    @style_loaded_output
    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None or self._result is None:
            return {}
        layer = context.getMapLayer(self.out_layer_id)
        if not layer:
            return {}
        result = self._result
        apply_output_metadata(layer, result.layer_title or self.DISPLAY, result.field_docs, self.DISPLAY)
        renderer = self._renderer_for(layer, result.style)
        if renderer is not None:
            apply_renderer(layer, renderer)
        return {}


def permutation_resolution_warning(permutations: int, tests: int, correction: str, two_sided: bool = True):
    """Say so when the permutation count cannot reach the corrected threshold.

    The smallest p-value P permutations can produce is 1/(P+1) one-sided and
    2/(P+1) two-sided.  Under Bonferroni a unit needs p < 0.05/m; under FDR
    the strongest units need roughly p < 0.05 k/m.  With too few permutations
    every unit is "not significant" for arithmetic reasons, which reads as a
    finding unless the report says otherwise.
    """
    floor = (2.0 if two_sided else 1.0) / (permutations + 1.0)
    if tests <= 0 or correction == "none":
        return ""
    if correction == "bonferroni" and floor >= 0.05 / tests:
        need = int((2.0 if two_sided else 1.0) * tests / 0.05)
        return (f"With {permutations} permutations the smallest attainable p-value is {floor:.4f}, which can never "
                f"pass a Bonferroni threshold over {tests} tests (0.05/{tests}). Use at least {need} permutations, "
                "or the FDR correction.")
    if correction == "fdr_bh" and floor * tests / max(1, tests // 10) >= 0.05:
        return (f"With {permutations} permutations the smallest attainable p-value is {floor:.4f}; over {tests} tests "
                "the FDR correction can only confirm units if a large share of them sit at that floor. A result with "
                "few or no significant units may reflect the permutation count rather than the data - re-run with "
                "999 or more permutations before concluding there is no local pattern.")
    return ""


def significance_label(p_value, alpha=0.05):
    return bool(p_value is not None and np.isfinite(p_value) and p_value < alpha)
