# -*- coding: utf-8 -*-
"""Bivariate Spatial Association (Lee's L) Processing Algorithm."""
from __future__ import annotations

import logging

import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsCategorizedSymbolRenderer,
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
    QgsProject,
    QgsRendererCategory,
    QgsSymbol,
)

from ..core.analysis_diagnostics import neighbor_summary, numeric_quality_summary, push_diagnostics
from ..core.stats_engines import calculate_bivariate_lee_l
from ..core.weights import build_weights_matrix

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class BivariateLeeLAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    X_FIELD = "X_FIELD"
    Y_FIELD = "Y_FIELD"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "bivariate_spatial_association_lees_l"

    def displayName(self) -> str:
        return "Bivariate Spatial Association (Lee's L)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("bivariate_spatial_association_lees_l")

    def createInstance(self):
        return BivariateLeeLAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Evaluates local bivariate spatial association between one feature attribute "
            "and the spatial lag of another attribute. Positive values indicate that high "
            "or low values of the first field are near similarly high or low neighboring "
            "values of the second field; negative values indicate local cross-variable contrast.\n\n"
            "The output includes lee_l, y_lag_z, lee_class, and lee_nbrs diagnostic fields. "
            "This implementation is a planning-oriented Lee's L style local diagnostic and "
            "does not run permutation inference."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.TypeVectorAnyGeometry],
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.X_FIELD,
                "Primary numeric field (X)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.Y_FIELD,
                "Neighbor-lag numeric field (Y)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial relationship / weights type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.KNN,
                "Number of neighbors (K value, KNN only)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=5,
                minValue=1,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=1000.0,
                minValue=0.0001,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output bivariate association layer"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")
        x_field = self.parameterAsString(parameters, self.X_FIELD, context)
        y_field = self.parameterAsString(parameters, self.Y_FIELD, context)
        if x_field == y_field:
            raise QgsProcessingException("Choose two different fields for bivariate association.")

        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_type = ["queen", "rook", "knn", "distance"][weight_type_idx]
        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)

        x_idx = source.fields().lookupField(x_field)
        y_idx = source.fields().lookupField(y_field)
        if x_idx < 0:
            raise QgsProcessingException(f"Field '{x_field}' not found.")
        if y_idx < 0:
            raise QgsProcessingException(f"Field '{y_field}' not found.")

        feedback.pushInfo("Building spatial weights for bivariate association...")
        neighbors, weights, id_order, _ = build_weights_matrix(
            source,
            weight_type,
            k_neighbors=k_neighbors,
            distance_band=distance_band,
            feedback=feedback,
        )

        x_values = {}
        y_values = {}
        total = source.featureCount() or 1
        for feature in source.getFeatures():
            x_val = self._to_float(feature.attribute(x_idx))
            y_val = self._to_float(feature.attribute(y_idx))
            if x_val is None or y_val is None:
                continue
            x_values[feature.id()] = x_val
            y_values[feature.id()] = y_val

        valid_id_order = [fid for fid in id_order if fid in x_values and fid in y_values]
        if len(valid_id_order) < 3:
            raise QgsProcessingException("At least 3 complete numeric records are required.")
        x_array = np.array([x_values[fid] for fid in valid_id_order], dtype=float)
        y_array = np.array([y_values[fid] for fid in valid_id_order], dtype=float)
        x_summary = numeric_quality_summary(total, x_values, x_array)
        n_summary = neighbor_summary(neighbors, valid_id_order)
        push_diagnostics(feedback, x_summary, n_summary, None)
        if x_summary["is_constant"] or np.std(y_array) == 0:
            raise QgsProcessingException("Both selected fields must vary across complete records.")

        lee_l, y_lag_z, classes = calculate_bivariate_lee_l(
            x_array, y_array, neighbors, weights, valid_id_order
        )

        out_fields = source.fields()
        out_fields.append(QgsField("lee_l", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("y_lag_z", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("lee_class", QVariant.String, len=32))
        out_fields.append(QgsField("lee_nbrs", QVariant.Int))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs(),
        )
        self.out_layer_id = dest_id
        valid_ids = set(valid_id_order)
        result_map = {}
        for idx, fid in enumerate(valid_id_order):
            neighbor_count = len([nid for nid in neighbors.get(fid, []) if nid in valid_ids])
            result_map[fid] = (lee_l[idx], y_lag_z[idx], classes[idx], neighbor_count)

        for current, feature in enumerate(source.getFeatures()):
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                l_value, lag_value, class_value, neighbor_count = result_map[fid]
                out_feature.setAttribute("lee_l", float(l_value))
                out_feature.setAttribute("y_lag_z", float(lag_value))
                out_feature.setAttribute("lee_class", class_value)
                out_feature.setAttribute("lee_nbrs", int(neighbor_count))
            else:
                out_feature.setAttribute("lee_l", None)
                out_feature.setAttribute("y_lag_z", None)
                out_feature.setAttribute("lee_class", None)
                out_feature.setAttribute("lee_nbrs", None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))
        return {self.OUTPUT: dest_id}

    def _to_float(self, value):
        if value is None or value == QVariant() or str(value) == "NULL":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}
        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}
        feedback.pushInfo("Applying Lee's L bivariate association styling...")
        palette = {
            "High-X / High-Y Lag": "#b2182b",
            "Low-X / Low-Y Lag": "#2166ac",
            "High-X / Low-Y Lag": "#ef8a62",
            "Low-X / High-Y Lag": "#67a9cf",
            "Not Significant": "#f7f7f7",
        }
        categories = []
        for value, color_hex in palette.items():
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)
            categories.append(QgsRendererCategory(value, symbol, value))
        layer.setRenderer(QgsCategorizedSymbolRenderer("lee_class", categories))
        layer.triggerRepaint()
        return {}
