# -*- coding: utf-8 -*-
"""Mean Center & Central Feature Processing Algorithm."""
from __future__ import annotations

import logging
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    NULL,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsPointXY,
    QgsGeometry,
    QgsProject,
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
from ..core.layer_metadata import apply_output_metadata
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class MeanCenterAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    WEIGHT_FIELD = "WEIGHT_FIELD"
    MODE = "MODE"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "mean_center"

    def displayName(self) -> str:
        # Avoid commercial trademarks
        return "Central Feature / Mean Center"

    def group(self) -> str:
        return "04 | Centers, Direction and Dispersion"

    def groupId(self) -> str:
        return "planx_center_direction_spread"

    def icon(self):
        return algorithm_icon("mean_center")

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
        skipped_geometry = 0
        invalid_weights = 0
        negative_weights = 0

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
            if geom is None or geom.isEmpty():
                skipped_geometry += 1
                continue

            centroid = geometry_centroid_point(geom)
            if centroid is None:
                skipped_geometry += 1
                continue
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())

            w_val = 1.0
            if has_weight:
                val = f.attribute(weight_field)
                if val is not None and val != NULL and str(val) != 'NULL':
                    try:
                        w_val = float(val)
                        if w_val < 0:
                            negative_weights += 1
                    except (ValueError, TypeError):
                        invalid_weights += 1
                        w_val = 1.0
                else:
                    invalid_weights += 1
            weights.append(w_val)
            features.append(f)
            feedback.setProgress(int(30 * (idx / total)))

        if len(features) == 0:
            raise QgsProcessingException("No valid geometries found in the input layer.")
        if negative_weights:
            raise QgsProcessingException("Weight field contains negative values. Use non-negative weights for center calculations.")

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)
        w_arr = np.array(weights)
        if has_weight and float(np.sum(w_arr)) <= 0:
            raise QgsProcessingException("The sum of weights must be greater than zero.")
        feedback.pushInfo(
            f"Center diagnostics: valid={len(features)}, skipped_geometry={skipped_geometry}, "
            f"invalid_or_null_weights={invalid_weights}, total_weight={float(np.sum(w_arr)):.6f}."
        )

        # Define outputs dynamically based on the mode
        if mode_idx == 0:
            # Mean Center: Output is a Point
            out_geom_type = QgsWkbTypes.Point
            out_fields = QgsFields()
            out_fields.append(QgsField("mean_x", QVariant.Double, len=15, prec=6))
            out_fields.append(QgsField("mean_y", QVariant.Double, len=15, prec=6))
            out_fields.append(QgsField("total_w", QVariant.Double, len=15, prec=6))
            out_fields.append(QgsField("input_n", QVariant.Int))
            out_fields.append(QgsField("skip_geom", QVariant.Int))
            out_fields.append(QgsField("bad_w", QVariant.Int))
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
        self.out_layer_id = dest_id

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
            out_feat.setAttribute("input_n", len(features))
            out_feat.setAttribute("skip_geom", skipped_geometry)
            out_feat.setAttribute("bad_w", invalid_weights)
            sink.addFeature(out_feat)
        else:
            # Central Feature
            cf_idx = calculate_central_feature(x_arr, y_arr, w_arr)
            cf = features[cf_idx]
            out_feat = QgsFeature(cf)
            sink.addFeature(out_feat)

        feedback.setProgress(100)
        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}
        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}
        apply_output_metadata(
            layer,
            "PlanX GeoStats mean center output",
            {
                "mean_x": "Mean center X coordinate",
                "mean_y": "Mean center Y coordinate",
                "total_w": "Total weight used in the mean center calculation",
                "input_n": "Number of valid input features used",
                "skip_geom": "Input features skipped because geometry was empty",
                "bad_w": "Input features with null or invalid weight values",
            },
            self.displayName(),
        )
        return {}
