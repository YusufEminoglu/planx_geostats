# -*- coding: utf-8 -*-
"""Getis-Ord Gi* Hotspot Analysis Processing Algorithm."""
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
from ..core.stats_engines import calculate_getis_ord

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class GetisOrdAlgorithm(QgsProcessingAlgorithm):
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
        return "getis_ord_gi"

    def displayName(self) -> str:
        return "Hot Spot Analysis (Getis-Ord Gi*)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("getis_ord_gi")

    def createInstance(self):
        return GetisOrdAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Given a set of weighted features, identifies statistically significant "
            "hot spots and cold spots using the Getis-Ord Gi* statistic.\n\n"
            "The output layer will include gi_zscore, gi_pvalue, gi_conf, and gi_nbrs. "
            "A confidence bin value of +3 indicates a 99% confidence hot spot, while -3 "
            "indicates a 99% confidence cold spot. The gi_nbrs field records how many "
            "valid neighboring features supported each local statistic."
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
                "Hot Spot analysis output layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        # Retrieve parameters
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

        # Build weights matrix
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

        if len(y) == 0:
            raise QgsProcessingException("No valid numeric values found in the target field.")

        feedback.pushInfo("Calculating Getis-Ord Gi* statistics...")
        z_scores, p_values, conf_bins = calculate_getis_ord(
            y,
            neighbors,
            weights,
            valid_id_order,
            star=True
        )

        if feedback.isCanceled():
            return {}

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("gi_zscore", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("gi_pvalue", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("gi_conf", QVariant.Int))
        out_fields.append(QgsField("gi_nbrs", QVariant.Int))

        # Initialize output sink
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
        valid_ids = set(valid_id_order)
        isolated_count = 0
        for idx, fid in enumerate(valid_id_order):
            neighbor_count = len([nid for nid in neighbors.get(fid, []) if nid in valid_ids])
            if neighbor_count == 0:
                isolated_count += 1
            results_map[fid] = (z_scores[idx], p_values[idx], conf_bins[idx], neighbor_count)
        if isolated_count:
            feedback.pushWarning(
                f"{isolated_count} feature(s) had no valid neighbors. Review gi_nbrs and consider a larger distance band or K value."
            )

        feedback.pushInfo("Writing results to output layer...")
        total = source.featureCount() or 1
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                z, p, c, neighbor_count = results_map[fid]
                out_feat.setAttribute("gi_zscore", float(z))
                out_feat.setAttribute("gi_pvalue", float(p))
                out_feat.setAttribute("gi_conf", int(c))
                out_feat.setAttribute("gi_nbrs", int(neighbor_count))
            else:
                out_feat.setAttribute("gi_zscore", None)
                out_feat.setAttribute("gi_pvalue", None)
                out_feat.setAttribute("gi_conf", None)
                out_feat.setAttribute("gi_nbrs", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(50 + 50 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        # Applies gorgeous hot/cold styling automatically in the QGIS GUI thread
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying Cold-to-Hot Hotspot symbology style...")

        categories = []
        # -3: 99% Cold, -2: 95% Cold, -1: 90% Cold, 0: Not Sig, 1: 90% Hot, 2: 95% Hot, 3: 99% Hot
        style_definitions = [
            (-3, '#2166ac', 'Cold Spot - 99% Confidence'),
            (-2, '#67a9cf', 'Cold Spot - 95% Confidence'),
            (-1, '#d1e5f0', 'Cold Spot - 90% Confidence'),
            (0, '#f7f7f7', 'Not Significant'),
            (1, '#fddbc7', 'Hot Spot - 90% Confidence'),
            (2, '#f4a582', 'Hot Spot - 95% Confidence'),
            (3, '#b2182b', 'Hot Spot - 99% Confidence')
        ]

        for val, color_hex, label in style_definitions:
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)

            # Fine tune borders/stroke colors safely for polygons and lines
            if symbol.symbolLayerCount() > 0:
                sl = symbol.symbolLayer(0)
                if hasattr(sl, 'setStrokeColor'):
                    sl.setStrokeColor(QColor('#a0a0a0'))
                if hasattr(sl, 'setStrokeWidth'):
                    sl.setStrokeWidth(0.1)
                if hasattr(sl, 'setOutlineColor'):
                    sl.setOutlineColor(QColor('#a0a0a0'))

            category = QgsRendererCategory(val, symbol, label, True)
            categories.append(category)

        renderer = QgsCategorizedSymbolRenderer('gi_conf', categories)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {}
