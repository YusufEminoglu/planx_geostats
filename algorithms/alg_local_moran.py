# -*- coding: utf-8 -*-
"""Anselin Local Moran's I (LISA) Processing Algorithm."""
from __future__ import annotations

import logging
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject,
    QgsFeature,
    QgsField,
    QgsSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterEnum,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsFeatureSink
)

from ..core.weights import build_weights_matrix
from ..core.stats_engines import calculate_local_moran

logger = logging.getLogger("PlanX-GeoStats")


class LocalMoranAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD = "FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "local_moran_lisa"

    def displayName(self) -> str:
        return "Cluster and Outlier Analysis (Local Moran's I)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def createInstance(self):
        return LocalMoranAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Given a set of weighted features, identifies statistically significant "
            "hot spots, cold spots, and spatial outliers using the Anselin Local Moran's I statistic.\n\n"
            "The output layer will include four new columns: lisa_i, lisa_z, lisa_p, "
            "and quadrant. The quadrants represent:\n"
            "- HH (High-High): High values surrounded by high values\n"
            "- LL (Low-Low): Low values surrounded by low values\n"
            "- HL (High-Low): Spatial outlier (high value surrounded by low values)\n"
            "- LH (Low-High): Spatial outlier (low value surrounded by high values)\n"
            "- Not Significant: Features that are not statistically significant (p >= 0.05)"
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
                self.FIELD,
                "Target numeric field to analyze",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Cluster Analysis Output Layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_name = self.parameterAsString(parameters, self.FIELD, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)

        # Validate target field
        field_idx = source.fields().lookupField(field_name)
        if field_idx < 0:
            raise QgsProcessingException(f"Target field '{field_name}' not found.")

        field = source.fields().at(field_idx)
        if not field.isNumeric():
            raise QgsProcessingException(f"Target field '{field_name}' must be numeric.")

        feedback.pushInfo("Generating spatial weights matrix...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source,
            weight_type,
            k_neighbors=k_neighbors,
            distance_band=distance_band,
            feedback=feedback
        )

        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Extracting target field values...")
        y_dict = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val = f.attribute(field_name)
            if val is None or val == QVariant() or str(val) == 'NULL':
                continue
            try:
                y_dict[f.id()] = float(val)
            except (ValueError, TypeError):
                continue

        # Filter id_order and construct y array
        valid_id_order = [fid for fid in id_order if fid in y_dict]
        y = np.array([y_dict[fid] for fid in valid_id_order])

        if len(y) <= 2:
            raise QgsProcessingException("At least 3 valid features with numeric values are required for LISA analysis.")

        feedback.pushInfo("Calculating Local Moran's I statistics...")
        i_vals, z_scores, p_values, quadrants = calculate_local_moran(
            y,
            neighbors,
            weights,
            valid_id_order
        )

        if feedback.isCanceled():
            return {}

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("lisa_i", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("lisa_z", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("lisa_p", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("quadrant", QVariant.String, len=20))

        # Setup sink
        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        self.out_layer_id = dest_id

        # Map results to feature IDs
        results_map = {}
        for idx, fid in enumerate(valid_id_order):
            results_map[fid] = (i_vals[idx], z_scores[idx], p_values[idx], quadrants[idx])

        feedback.pushInfo("Writing results to output layer...")
        total = source.featureCount() or 1
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                i_val, z, p, quad = results_map[fid]
                out_feat.setAttribute("lisa_i", float(i_val))
                out_feat.setAttribute("lisa_z", float(z))
                out_feat.setAttribute("lisa_p", float(p))
                out_feat.setAttribute("quadrant", str(quad))
            else:
                out_feat.setAttribute("lisa_i", None)
                out_feat.setAttribute("lisa_z", None)
                out_feat.setAttribute("lisa_p", None)
                out_feat.setAttribute("quadrant", "Not Significant")

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(50 + 50 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying LISA cluster analysis styling...")
        categories = []
        # HH: Red, LL: Blue, HL: Pink, LH: Light Blue, Not Significant: Gray
        style_definitions = [
            ('HH', '#e31a1c', 'High-High (HH)'),
            ('LL', '#1f78b4', 'Low-Low (LL)'),
            ('HL', '#fb9a99', 'High-Low (HL) outlier'),
            ('LH', '#a6cee3', 'Low-High (LH) outlier'),
            ('Not Significant', '#f7f7f7', 'Not Significant')
        ]

        for val, color_hex, label in style_definitions:
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)

            if symbol.symbolLayerCount() > 0:
                sl = symbol.symbolLayer(0)
                if hasattr(sl, 'setStrokeColor'):
                    sl.setStrokeColor(QColor('#b0b0b0'))
                if hasattr(sl, 'setStrokeWidth'):
                    sl.setStrokeWidth(0.1)
                if hasattr(sl, 'setOutlineColor'):
                    sl.setOutlineColor(QColor('#b0b0b0'))

            category = QgsRendererCategory(val, symbol, label, True)
            categories.append(category)

        renderer = QgsCategorizedSymbolRenderer('quadrant', categories)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {}
