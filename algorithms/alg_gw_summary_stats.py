# -*- coding: utf-8 -*-
"""Geographically Weighted Summary Statistics Processing Algorithm."""
from __future__ import annotations

import logging

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.stats_engines import calculate_gw_summary_stats
from ..core.symbology import apply_renderer, sequential_quantile_renderer
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class GWSummaryStatsAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    KERNEL_TYPE = "KERNEL_TYPE"
    BANDWIDTH = "BANDWIDTH"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "gw_summary_statistics"

    def displayName(self) -> str:
        return "Geographically Weighted Summary Statistics"

    def group(self) -> str:
        return "05 | Models and Scenarios"

    def groupId(self) -> str:
        return "planx_model_scenario"

    def icon(self):
        return algorithm_icon("gw_summary_statistics")

    def createInstance(self):
        return GWSummaryStatsAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes a local (kernel-weighted) mean, standard deviation, "
            "and skewness of one numeric field at every feature, using the "
            "same kernel families as GWR and MGWR (Fixed Gaussian, Fixed "
            "Bisquare, Adaptive Bisquare) - a spatial analog of a rolling "
            "average that answers 'what does this field look like around "
            "here', as distinct from Multiscale Geographically Weighted "
            "Regression's single global bandwidth or a plain global "
            "mean/std that hides all spatial variation.\n\n"
            "Output: the input layer with gw_mean, gw_std, gw_skew, and "
            "gw_neff (Kish effective sample size - roughly how many nearby "
            "observations are really informing this point's local "
            "statistics, given the kernel's soft weighting; a low gw_neff "
            "means the local estimate rests on very few effective "
            "observations and should be trusted less).\n\n"
            "Bandwidth has the same meaning and same too-small/too-large "
            "trade-off as in GWR: a distance in map units for the two fixed "
            "kernels, or a neighbor count for Adaptive Bisquare. Too small "
            "produces noisy, erratic local statistics with low gw_neff "
            "everywhere; too large approaches the plain global mean/std and "
            "erases the spatial variation you are trying to see.\n\n"
            "Use this to check the Constant-variance assumption behind "
            "global regression tools (map gw_std - if it varies sharply "
            "across the study area, a single global model may fit some "
            "areas much worse than others), or simply to smooth a noisy "
            "field for exploratory mapping before deciding whether it needs "
            "a full GWR/MGWR model."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD, "Field to summarize (numeric)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.KERNEL_TYPE, "Kernel type", options=["Fixed Gaussian", "Fixed Bisquare", "Adaptive Bisquare"], defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BANDWIDTH, "Bandwidth value (distance in map units for Fixed; number of neighbors for Adaptive)",
                type=QgsProcessingParameterNumber.Double, defaultValue=15.0, minValue=1.0,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output layer with local summary statistics"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        kernel_types = ["fixed_gaussian", "fixed_bisquare", "adaptive_bisquare"]
        kernel_type = kernel_types[self.parameterAsEnum(parameters, self.KERNEL_TYPE, context)]
        bandwidth = self.parameterAsDouble(parameters, self.BANDWIDTH, context)

        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Field '{field_name}' not found.")
        if kernel_type == "adaptive_bisquare" and bandwidth < 2:
            raise QgsProcessingException("Adaptive Bisquare bandwidth (neighbor count) must be at least 2.")

        coords, values, valid_fids, skipped = [], [], [], 0
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting complete numeric records and centroids...")
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            centroid = geometry_centroid_point(feature.geometry())
            raw = feature.attribute(field_idx)
            if centroid is None or raw is None or raw == NULL or str(raw) == "NULL":
                skipped += 1
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                skipped += 1
                continue
            if not np.isfinite(value):
                skipped += 1
                continue
            coords.append([centroid.x(), centroid.y()])
            values.append(value)
            valid_fids.append(feature.id())
            feedback.setProgress(int(30 * (idx / total)))

        if len(values) < 4:
            raise QgsProcessingException(f"Insufficient complete records ({len(values)}); at least 4 are required.")

        coords_arr = np.array(coords, dtype=float)
        values_arr = np.array(values, dtype=float)
        feedback.pushInfo(f"Computing geographically weighted summary statistics ({kernel_type}, bandwidth={bandwidth})...")
        results = calculate_gw_summary_stats(values_arr, coords_arr, bandwidth, kernel_type)
        feedback.pushInfo(
            f"Median gw_neff = {float(np.nanmedian(results['effective_n'])):.2f}; "
            f"mean gw_std = {float(np.nanmean(results['local_std'])):.4f}"
        )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing/invalid geometry or values.")

        out_fields = source.fields()
        out_fields.append(QgsField("gw_mean", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("gw_std", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("gw_skew", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("gw_neff", QVariant.Double, len=10, prec=2))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        result_map = {fid: idx for idx, fid in enumerate(valid_fids)}
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                row_idx = result_map[fid]
                out_feature.setAttribute("gw_mean", float(results["local_mean"][row_idx]))
                out_feature.setAttribute("gw_std", float(results["local_std"][row_idx]))
                out_feature.setAttribute("gw_skew", float(results["local_skew"][row_idx]))
                out_feature.setAttribute("gw_neff", float(results["effective_n"][row_idx]))
            else:
                out_feature.setAttribute("gw_mean", None)
                out_feature.setAttribute("gw_std", None)
                out_feature.setAttribute("gw_skew", None)
                out_feature.setAttribute("gw_neff", None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats geographically weighted summary statistics output",
            {
                "gw_mean": "Kernel-weighted local mean of the summarized field",
                "gw_std": "Kernel-weighted local standard deviation",
                "gw_skew": "Kernel-weighted local skewness",
                "gw_neff": "Kish effective sample size of the local kernel window (lower = fewer effective nearby observations)",
            },
            "gw_summary_statistics",
        )
        apply_renderer(layer, sequential_quantile_renderer(layer, layer.geometryType(), "gw_std"))
        return {}
