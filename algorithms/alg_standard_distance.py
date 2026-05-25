# -*- coding: utf-8 -*-
"""Standard Distance Processing Algorithm."""
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

from ..core.stats_engines import calculate_standard_distance
from ..core.layer_metadata import apply_output_metadata
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class StandardDistanceAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    WEIGHT_FIELD = "WEIGHT_FIELD"
    MULTIPLIER = "MULTIPLIER"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "standard_distance"

    def displayName(self) -> str:
        return "Standard Distance"

    def group(self) -> str:
        return "04 | Centers, Direction and Dispersion"

    def groupId(self) -> str:
        return "planx_center_direction_spread"

    def icon(self):
        return algorithm_icon("standard_distance")

    def createInstance(self):
        return StandardDistanceAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Measures the degree to which features are concentrated or dispersed "
            "around their geometric mean center by calculating the Standard Distance.\n\n"
            "Creates a circular polygon representing the standard distance. "
            "You can choose a multiplier of 1, 2, or 3 standard deviations."
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
                self.MULTIPLIER,
                "Circle size (standard deviations)",
                options=["1 standard deviation", "2 standard deviations", "3 standard deviations"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output standard distance layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        weight_field = self.parameterAsString(parameters, self.WEIGHT_FIELD, context)
        mult_idx = self.parameterAsEnum(parameters, self.MULTIPLIER, context)
        multiplier = mult_idx + 1  # 1, 2, or 3

        # Extract features and coordinates
        x_coords = []
        y_coords = []
        weights = []
        skipped_geometry = 0
        invalid_weights = 0
        negative_weights = 0

        has_weight = weight_field != ""
        if has_weight:
            field_idx = source.fields().lookupField(weight_field)
            if field_idx < 0:
                raise QgsProcessingException(f"Weight field '{weight_field}' not found.")

        feedback.pushInfo("Extracting coordinates...")
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
            feedback.setProgress(int(40 * (idx / total)))

        if len(x_coords) <= 1:
            raise QgsProcessingException("At least 2 valid features with geometries are required to calculate standard distance.")
        if negative_weights:
            raise QgsProcessingException("Weight field contains negative values. Use non-negative weights for standard distance.")

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)
        w_arr = np.array(weights)
        if has_weight and float(np.sum(w_arr)) <= 0:
            raise QgsProcessingException("The sum of weights must be greater than zero.")
        feedback.pushInfo(
            f"Standard distance diagnostics: valid={len(x_coords)}, skipped_geometry={skipped_geometry}, "
            f"invalid_or_null_weights={invalid_weights}, total_weight={float(np.sum(w_arr)):.6f}."
        )

        feedback.pushInfo("Calculating standard distance...")
        mean_x, mean_y, std_dist = calculate_standard_distance(x_arr, y_arr, w_arr)

        radius = std_dist * multiplier

        # Construct circle polygon
        feedback.pushInfo("Constructing circle polygon...")
        points = []
        num_points = 72
        for i in range(num_points):
            alpha = (i / num_points) * 2.0 * np.pi
            x_c = mean_x + radius * np.cos(alpha)
            y_c = mean_y + radius * np.sin(alpha)
            points.append(QgsPointXY(x_c, y_c))
        points.append(points[0])  # close ring

        # Prepare fields
        out_fields = QgsFields()
        out_fields.append(QgsField("mean_x", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("mean_y", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("std_dist", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("multiplier", QVariant.Int))
        out_fields.append(QgsField("radius", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("input_n", QVariant.Int))
        out_fields.append(QgsField("skip_geom", QVariant.Int))
        out_fields.append(QgsField("bad_w", QVariant.Int))

        # Setup sink
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            QgsWkbTypes.Polygon,
            source.sourceCrs()
        )
        self.out_layer_id = dest_id

        out_feat = QgsFeature()
        out_feat.setGeometry(QgsGeometry.fromPolygonXY([points]))
        out_feat.setFields(out_fields)
        out_feat.setAttribute("mean_x", mean_x)
        out_feat.setAttribute("mean_y", mean_y)
        out_feat.setAttribute("std_dist", std_dist)
        out_feat.setAttribute("multiplier", multiplier)
        out_feat.setAttribute("radius", radius)
        out_feat.setAttribute("input_n", len(x_coords))
        out_feat.setAttribute("skip_geom", skipped_geometry)
        out_feat.setAttribute("bad_w", invalid_weights)

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
            "PlanX GeoStats standard distance output",
            {
                "mean_x": "Mean center X coordinate for the standard distance circle",
                "mean_y": "Mean center Y coordinate for the standard distance circle",
                "std_dist": "One-standard-distance radius before multiplier",
                "multiplier": "Selected standard-distance multiplier",
                "radius": "Output circle radius in layer map units",
                "input_n": "Number of valid input features used",
                "skip_geom": "Input features skipped because geometry was empty",
                "bad_w": "Input features with null or invalid weight values",
            },
            self.displayName(),
        )
        return {}
