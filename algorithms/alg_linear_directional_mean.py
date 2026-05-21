# -*- coding: utf-8 -*-
"""Linear Directional Mean Processing Algorithm."""
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
    QgsWkbTypes,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink
)

from ..core.stats_engines import calculate_linear_directional_mean

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class LinearDirectionalMeanAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        return "linear_directional_mean"

    def displayName(self) -> str:
        return "Linear Directional Mean"

    def group(self) -> str:
        return "04 | Centers, Direction and Dispersion"

    def groupId(self) -> str:
        return "planx_center_direction_spread"

    def icon(self):
        return algorithm_icon("linear_directional_mean")

    def createInstance(self):
        return LinearDirectionalMeanAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Identifies the mean direction, mean length, and geographic center for "
            "a set of line features.\n\n"
            "Computes the circular weighted mean of line orientations (weighted by line "
            "length) and outputs a single representative trend line centered at the "
            "geographic mean of all line midpoints."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input line vector layer",
                [QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output directional mean line"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        # Extract line start/end coordinates
        s_x, s_y, e_x, e_y = [], [], [], []

        feedback.pushInfo("Extracting line endpoints...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom = f.geometry()
            if geom.isEmpty() or geom.type() != QgsWkbTypes.LineGeometry:
                continue

            # Get the first polyline part
            if geom.isMultipart():
                parts = geom.asMultiPolyline()
                if not parts or not parts[0]:
                    continue
                line = parts[0]
            else:
                line = geom.asPolyline()
                if not line:
                    continue

            start_pt = line[0]
            end_pt = line[-1]

            s_x.append(start_pt.x())
            s_y.append(start_pt.y())
            e_x.append(end_pt.x())
            e_y.append(end_pt.y())

            feedback.setProgress(int(50 * (idx / total)))

        n_lines = len(s_x)
        if n_lines == 0:
            raise QgsProcessingException("No valid line features found in the input layer.")

        feedback.pushInfo(f"Processing {n_lines} line features...")

        center_x, center_y, mean_angle, mean_length = calculate_linear_directional_mean(
            np.array(s_x), np.array(s_y),
            np.array(e_x), np.array(e_y)
        )

        feedback.pushInfo(f"Mean Angle: {mean_angle:.2f} degrees  |  Mean Length: {mean_length:.4f}  |  Center: ({center_x:.4f}, {center_y:.4f})")

        # Build the output trend line
        angle_rad = math.radians(mean_angle)
        half_len = mean_length / 2.0

        # Compass bearing: dx = sin(angle), dy = cos(angle)
        dx = half_len * math.sin(angle_rad)
        dy = half_len * math.cos(angle_rad)

        pt_start = QgsPointXY(center_x - dx, center_y - dy)
        pt_end = QgsPointXY(center_x + dx, center_y + dy)

        out_fields = QgsFields()
        out_fields.append(QgsField("mean_angle", QVariant.Double, len=10, prec=4))
        out_fields.append(QgsField("mean_length", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("center_x", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("center_y", QVariant.Double, len=15, prec=6))
        out_fields.append(QgsField("line_count", QVariant.Int))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            QgsWkbTypes.LineString,
            source.sourceCrs()
        )

        out_feat = QgsFeature()
        out_feat.setGeometry(QgsGeometry.fromPolylineXY([pt_start, pt_end]))
        out_feat.setFields(out_fields)
        out_feat.setAttribute("mean_angle", mean_angle)
        out_feat.setAttribute("mean_length", mean_length)
        out_feat.setAttribute("center_x", center_x)
        out_feat.setAttribute("center_y", center_y)
        out_feat.setAttribute("line_count", n_lines)

        sink.addFeature(out_feat)
        feedback.setProgress(100)

        return {self.OUTPUT: dest_id}
