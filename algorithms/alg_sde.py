# -*- coding: utf-8 -*-
"""Standard Deviational Ellipse (SDE) Processing Algorithm."""
from __future__ import annotations

import logging
import math
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
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

from ..core.stats_engines import calculate_sde
from ..core.layer_metadata import apply_output_metadata
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class SDEAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    WEIGHT_FIELD = "WEIGHT_FIELD"
    STD_DEV = "STD_DEV"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "directional_distribution"

    def displayName(self) -> str:
        return "Directional Distribution (Standard Deviational Ellipse)"

    def group(self) -> str:
        return "04 | Centers, Direction and Dispersion"

    def groupId(self) -> str:
        return "planx_center_direction_spread"

    def icon(self):
        return algorithm_icon("directional_distribution")

    def createInstance(self):
        return SDEAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Measures the directional distribution of a set of features "
            "by calculating the Standard Deviational Ellipse (SDE).\n\n"
            "Creates a polygon ellipse that represents the spatial dispersion "
            "and orientation of features. You can choose to calculate 1, 2, "
            "or 3 standard deviations."
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
                self.STD_DEV,
                "Ellipse size (standard deviations)",
                options=["1 standard deviation", "2 standard deviations", "3 standard deviations"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output Ellipse Layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        weight_field = self.parameterAsString(parameters, self.WEIGHT_FIELD, context)
        std_dev_idx = self.parameterAsEnum(parameters, self.STD_DEV, context)
        num_std = std_dev_idx + 1  # 1, 2, or 3

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
                if val is not None and val != QVariant() and str(val) != 'NULL':
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
            feedback.setProgress(int(30 * (idx / total)))

        if len(x_coords) <= 2:
            raise QgsProcessingException("At least 3 valid features with geometries are required to calculate SDE.")
        if negative_weights:
            raise QgsProcessingException("Weight field contains negative values. Use non-negative weights for SDE.")

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)
        w_arr = np.array(weights)
        if has_weight and float(np.sum(w_arr)) <= 0:
            raise QgsProcessingException("The sum of weights must be greater than zero.")
        feedback.pushInfo(
            f"SDE diagnostics: valid={len(x_coords)}, skipped_geometry={skipped_geometry}, "
            f"invalid_or_null_weights={invalid_weights}, total_weight={float(np.sum(w_arr)):.6f}."
        )

        feedback.pushInfo("Calculating SDE parameters...")
        mean_x, mean_y, angle_rad, semi_major, semi_minor = calculate_sde(x_arr, y_arr, w_arr, num_std)

        # Construct ellipse polygon
        feedback.pushInfo("Constructing ellipse polygon...")
        points = []
        num_points = 72  # Every 5 degrees for a smooth polygon
        for i in range(num_points):
            alpha = (i / num_points) * 2.0 * np.pi
            
            # Point on ellipse relative to center (unrotated)
            x_ell = semi_major * np.cos(alpha)
            y_ell = semi_minor * np.sin(alpha)
            
            # Rotate
            x_rot = x_ell * np.cos(angle_rad) - y_ell * np.sin(angle_rad)
            y_rot = x_ell * np.sin(angle_rad) + y_ell * np.cos(angle_rad)
            
            # Translate to mean center
            x_final = x_rot + mean_x
            y_final = y_rot + mean_y
            
            points.append(QgsPointXY(x_final, y_final))
            
        points.append(points[0])  # Close the ring

        # Prepare fields
        out_fields = QgsFields()
        out_fields.append(QgsField("mean_x", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("mean_y", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("rotation", QVariant.Double, len=10, prec=4))
        out_fields.append(QgsField("semi_major", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("semi_minor", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("std_dev", QVariant.Int))
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
        # Convert rotation to degrees for presentation
        out_feat.setAttribute("rotation", float(math.degrees(angle_rad)))
        out_feat.setAttribute("semi_major", semi_major)
        out_feat.setAttribute("semi_minor", semi_minor)
        out_feat.setAttribute("std_dev", num_std)
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
            "PlanX GeoStats directional distribution output",
            {
                "mean_x": "Ellipse center X coordinate",
                "mean_y": "Ellipse center Y coordinate",
                "rotation": "Ellipse rotation angle in degrees",
                "semi_major": "Semi-major axis length in layer map units",
                "semi_minor": "Semi-minor axis length in layer map units",
                "std_dev": "Selected standard-deviation multiplier",
                "input_n": "Number of valid input features used",
                "skip_geom": "Input features skipped because geometry was empty",
                "bad_w": "Input features with null or invalid weight values",
            },
            self.displayName(),
        )
        return {}
