# -*- coding: utf-8 -*-
"""Central Feature Processing Algorithm."""
from __future__ import annotations

import logging
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    NULL,
    QgsFeature,
    QgsField,
    QgsProject,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterFeatureSink,
    QgsFeatureSink
)

from ..core.stats_engines import calculate_central_feature
from ..core.layer_metadata import apply_output_metadata
from ..core.weights import geometry_centroid_point

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class CentralFeatureAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    WEIGHT_FIELD = "WEIGHT_FIELD"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "central_feature"

    def displayName(self) -> str:
        return "Central Feature"

    def group(self) -> str:
        return "04 | Centers, Direction and Dispersion"

    def groupId(self) -> str:
        return "planx_center_direction_spread"

    def icon(self):
        return algorithm_icon("central_feature")

    def createInstance(self):
        return CentralFeatureAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Identifies the most centrally located feature in a dataset.\n\n"
            "The central feature is the one that minimizes the total Euclidean "
            "distance to all other features. An optional weight field can be used "
            "to emphasize certain features in the distance calculation."
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
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output central feature layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        weight_field = self.parameterAsString(parameters, self.WEIGHT_FIELD, context)

        x_coords = []
        y_coords = []
        weights = []
        features_list = []

        has_weight = weight_field != ""

        feedback.pushInfo("Extracting feature centroids...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom is None or geom.isEmpty():
                continue

            centroid = geometry_centroid_point(geom)
            if centroid is None:
                continue
            x_coords.append(centroid.x())
            y_coords.append(centroid.y())
            features_list.append(f)

            w_val = 1.0
            if has_weight:
                val = f.attribute(weight_field)
                if val is not None and val != NULL and str(val) != 'NULL':
                    try:
                        w_val = float(val)
                    except (ValueError, TypeError):
                        pass
            weights.append(w_val)
            feedback.setProgress(int(50 * (idx / total)))

        n = len(x_coords)
        if n == 0:
            raise QgsProcessingException("No features with valid geometries found.")

        feedback.pushInfo(f"Computing central feature among {n} features...")
        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)
        w_arr = np.array(weights) if has_weight else None

        central_idx = calculate_central_feature(x_arr, y_arr, w_arr)
        central_feat = features_list[central_idx]

        # Output fields: original fields + central_feature flag + total_distance
        out_fields = source.fields()
        out_fields.append(QgsField("is_central", QVariant.Int))
        out_fields.append(QgsField("total_distance", QVariant.Double, len=15, prec=6))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        self.out_layer_id = dest_id

        # Calculate total distance for the central feature
        coords = np.column_stack((x_arr, y_arr))
        dists = np.sqrt(np.sum((coords - coords[central_idx]) ** 2, axis=1))
        if w_arr is not None:
            total_dist = float(np.sum(dists * w_arr))
        else:
            total_dist = float(np.sum(dists))

        out_feat = QgsFeature(central_feat)
        out_feat.setFields(out_fields)
        out_feat.setAttribute("is_central", 1)
        out_feat.setAttribute("total_distance", total_dist)

        sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
        feedback.setProgress(100)

        feedback.pushInfo(
            f"Central feature ID: {central_feat.id()} | "
            f"Total distance: {total_dist:.4f}"
        )

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}
        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}
        apply_output_metadata(
            layer,
            "PlanX GeoStats central feature output",
            {
                "is_central": "1 for the selected central feature",
                "total_distance": "Total weighted distance from this feature to all valid input features",
            },
            self.displayName(),
        )
        return {}
