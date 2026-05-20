# -*- coding: utf-8 -*-
"""Median Center Processing Algorithm."""
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
    QgsProcessingParameterFeatureSink
)

from ..core.stats_engines import calculate_median_center

logger = logging.getLogger("PlanX GeoStats Lab")


class MedianCenterAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    WEIGHT_FIELD = "WEIGHT_FIELD"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "median_center"

    def displayName(self) -> str:
        return "Median Center"

    def group(self) -> str:
        return "04 | Centers, Direction and Dispersion"

    def groupId(self) -> str:
        return "planx_center_direction_spread"

    def createInstance(self):
        return MedianCenterAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes the Median Center, which is the coordinate location that minimizes "
            "the sum of Euclidean distances from all features to that point.\n\n"
            "This algorithm iteratively solves for the optimal center using Weiszfeld's "
            "algorithm, and outputs a single point feature layer."
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
                "Output median center layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        weight_field = self.parameterAsString(parameters, self.WEIGHT_FIELD, context)

        # Extract features
        x_coords = []
        y_coords = []
        weights = []

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
            feedback.setProgress(int(40 * (idx / total)))

        if len(x_coords) == 0:
            raise QgsProcessingException("No features with valid geometries were found.")

        x_arr = np.array(x_coords)
        y_arr = np.array(y_coords)
        w_arr = np.array(weights)

        feedback.pushInfo("Iterating using Weiszfeld's algorithm...")
        med_x, med_y, total_dist = calculate_median_center(x_arr, y_arr, w_arr)

        # Output fields
        out_fields = QgsFields()
        out_fields.append(QgsField("median_x", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("median_y", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("total_dist", QVariant.Double, len=15, prec=6))

        # Setup sink
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            QgsWkbTypes.Point,
            source.sourceCrs()
        )

        out_feat = QgsFeature()
        out_feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(med_x, med_y)))
        out_feat.setFields(out_fields)
        out_feat.setAttribute("median_x", med_x)
        out_feat.setAttribute("median_y", med_y)
        out_feat.setAttribute("total_dist", total_dist)

        sink.addFeature(out_feat)
        feedback.setProgress(100)

        return {self.OUTPUT: dest_id}
