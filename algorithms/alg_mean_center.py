# -*- coding: utf-8 -*-
"""Mean Center & Central Feature Processing Algorithm."""
from __future__ import annotations

import logging
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFields,
    QgsPointXY,
    QgsGeometry,
    QgsWkbTypes,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink
)

from ..core.stats_engines import calculate_mean_center, calculate_central_feature

logger = logging.getLogger("PlanX-GeoStats")


class MeanCenterAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    WEIGHT_FIELD = "WEIGHT_FIELD"
    MODE = "MODE"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "mean_center"

    def displayName(self) -> str:
        # Avoid commercial trademarks
        return "Central Feature / Mean Center"

    def group(self) -> str:
        return "04 | Centers, Direction and Dispersion"

    def groupId(self) -> str:
        return "planx_center_direction_spread"

    def createInstance(self):
        return MeanCenterAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Calculates the geographic center of a set of features.\n\n"
            "Options:\n"
            "1. Mean Center: Computes the average coordinate centroid, "
            "optionally weighted by a numeric attribute.\n"
            "2. Central Feature: Finds the individual feature that has "
            "the shortest cumulative distance to all other features in the layer."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.WEIGHT_FIELD,
                "Weight field (optional)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                optional=True
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.MODE,
                "Center calculation mode",
                options=["Mean Center", "Central Feature"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        weight_field = self.parameterAsString(parameters, self.WEIGHT_FIELD, context)
        mode_idx = self.parameterAsEnum(parameters, self.MODE, context)

        # Extract features and coordinates
        x_coords = []
        y_coords = []
        weights = []
        features = []

        has_weight = weight_field != ""
        if has_weight:
            # Validate field exists
            field_idx = source.fields().lookupField(weight_field)
            if field_idx < 0:
                raise QgsProcessingException(f"Weight field '{weight_field}' not found.")

        feedback.pushInfo("Extracting coordinate geometries...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom.isEmpty():
                continue

            centroid = geom.centroid().asPoint()
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())

            w_val = 1.0
            if has_weight:
                val = f.attribute(weight_field)
                if val is not None and val != QVariant() and str(val) != 'NULL':
                    try:
                        w_val = float(val)
                    except (ValueError, TypeError):
                        pass
            weights.append(w_val)
            features.append(f)
            feedback.setProgress(int(30 * (idx / total)))

        if len(features) == 0:
            raise QgsProcessingException("No valid geometries found in the input layer.")

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)
        w_arr = np.array(weights)

        # Define outputs dynamically based on the mode
        if mode_idx == 0:
            # Mean Center: Output is a Point
            out_geom_type = QgsWkbTypes.Point
            out_fields = QgsFields()
            out_fields.append(QgsField("mean_x", QVariant.Double, len=15, prec=6))
            out_fields.append(QgsField("mean_y", QVariant.Double, len=15, prec=6))
            out_fields.append(QgsField("total_w", QVariant.Double, len=15, prec=6))
        else:
            # Central Feature: Output geometry and fields are identical to source
            out_geom_type = source.wkbType()
            out_fields = source.fields()

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            out_geom_type,
            source.sourceCrs()
        )

        feedback.pushInfo("Calculating geographic center...")
        if mode_idx == 0:
            # Mean Center
            mean_x, mean_y = calculate_mean_center(x_arr, y_arr, w_arr)
            out_feat = QgsFeature()
            out_feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(mean_x, mean_y)))
            out_feat.setFields(out_fields)
            out_feat.setAttribute("mean_x", mean_x)
            out_feat.setAttribute("mean_y", mean_y)
            out_feat.setAttribute("total_w", float(np.sum(w_arr)))
            sink.addFeature(out_feat)
        else:
            # Central Feature
            cf_idx = calculate_central_feature(x_arr, y_arr, w_arr)
            cf = features[cf_idx]
            out_feat = QgsFeature(cf)
            sink.addFeature(out_feat)

        feedback.setProgress(100)
        return {self.OUTPUT: dest_id}
