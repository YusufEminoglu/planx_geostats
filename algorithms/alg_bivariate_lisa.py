# -*- coding: utf-8 -*-
"""Bivariate Local Moran's I (Bivariate LISA) Processing Algorithm."""
from __future__ import annotations

import logging
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
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
from ..core.stats_engines import calculate_bivariate_local_moran
from ..core.layer_metadata import apply_output_metadata
from ..core.local_pattern_audit import local_moran_class_summary

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class BivariateLISAAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELD_X = "FIELD_X"
    FIELD_Y = "FIELD_Y"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    PERMUTATIONS = "PERMUTATIONS"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "bivariate_lisa"

    def displayName(self) -> str:
        return "Bivariate Cluster and Outlier Analysis (Bivariate LISA)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("bivariate_lisa")

    def createInstance(self):
        return BivariateLISAAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Extends Local Moran's I to two variables: for each feature, tests "
            "whether variable X at that location is significantly associated "
            "with the spatial lag (neighborhood average) of variable Y, using "
            "conditional-permutation inference (999 permutations by default, "
            "configurable 99-9999) rather than an analytical formula - the "
            "bivariate case has no simple closed-form null distribution.\n\n"
            "Output fields: bilisa_i, bilisa_z, bilisa_p, bilisa_nb (valid "
            "neighbor count), and quadrant, assigned when bilisa_p < 0.05:\n"
            "- HH (High-High): high X surrounded by high neighboring Y\n"
            "- LL (Low-Low): low X surrounded by low neighboring Y\n"
            "- HL (High-Low): high X surrounded by low neighboring Y\n"
            "- LH (Low-High): low X surrounded by high neighboring Y\n\n"
            "Bivariate LISA is directional and asymmetric: X's local relationship "
            "to lagged Y is not the same computation as Y's local relationship to "
            "lagged X, so running the tool with the fields swapped can produce a "
            "genuinely different map, not just a relabeled one - run both "
            "directions when the causal direction between X and Y is unclear. As "
            "with any bivariate spatial statistic, a significant result reflects "
            "association, not causation, and can be inflated by each variable's "
            "own spatial autocorrelation even when no real cross-variable "
            "relationship exists; check univariate Local Moran's I on X and Y "
            "separately as a sanity check."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Input vector layer",
                [QgsProcessing.SourceType.TypeVectorAnyGeometry]
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_X,
                "First numeric field (Variable X)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.DataType.Numeric
            )
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELD_Y,
                "Second numeric field (Variable Y)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.DataType.Numeric
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
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=5,
                minValue=1
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.DISTANCE_BAND,
                "Distance band threshold (map units, Distance Band only)",
                type=QgsProcessingParameterNumber.Type.Double,
                defaultValue=1000.0,
                minValue=0.0001
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PERMUTATIONS,
                "Number of permutations (Monte Carlo)",
                type=QgsProcessingParameterNumber.Type.Integer,
                defaultValue=999,
                minValue=99,
                maxValue=9999
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Bivariate LISA Output Layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        field_x_name = self.parameterAsString(parameters, self.FIELD_X, context)
        field_y_name = self.parameterAsString(parameters, self.FIELD_Y, context)
        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]

        k_neighbors = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        perms = self.parameterAsInt(parameters, self.PERMUTATIONS, context)

        # Validate target fields
        field_x_idx = source.fields().lookupField(field_x_name)
        if field_x_idx < 0:
            raise QgsProcessingException(f"Variable X field '{field_x_name}' not found.")
        field_x = source.fields().at(field_x_idx)
        if not field_x.isNumeric():
            raise QgsProcessingException(f"Variable X field '{field_x_name}' must be numeric.")

        field_y_idx = source.fields().lookupField(field_y_name)
        if field_y_idx < 0:
            raise QgsProcessingException(f"Variable Y field '{field_y_name}' not found.")
        field_y = source.fields().at(field_y_idx)
        if not field_y.isNumeric():
            raise QgsProcessingException(f"Variable Y field '{field_y_name}' must be numeric.")

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
        x_dict = {}
        y_dict = {}
        for f in source.getFeatures():
            if feedback.isCanceled():
                break
            val_x = f.attribute(field_x_name)
            val_y = f.attribute(field_y_name)
            if val_x in (None, NULL) or val_y in (None, NULL):
                continue
            try:
                x_dict[f.id()] = float(val_x)
                y_dict[f.id()] = float(val_y)
            except (ValueError, TypeError):
                continue

        # Filter id_order and construct x, y arrays
        valid_id_order = [fid for fid in id_order if fid in x_dict and fid in y_dict]
        x_arr = np.array([x_dict[fid] for fid in valid_id_order])
        y_arr = np.array([y_dict[fid] for fid in valid_id_order])

        if len(x_arr) <= 2:
            raise QgsProcessingException("At least 3 valid features with numeric values are required for LISA analysis.")

        feedback.pushInfo(f"Calculating Bivariate Local Moran's I statistics using {perms} permutations...")
        i_vals, z_scores, p_values, quadrants = calculate_bivariate_local_moran(
            x_arr,
            y_arr,
            neighbors,
            weights,
            valid_id_order,
            permutations=perms
        )
        class_summary = local_moran_class_summary(quadrants)
        feedback.pushInfo(class_summary["message"])

        if feedback.isCanceled():
            return {}

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("bilisa_i", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("bilisa_z", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("bilisa_p", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("quadrant", QVariant.String, len=20))
        out_fields.append(QgsField("bilisa_nb", QVariant.Int))

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
        valid_ids = set(valid_id_order)
        isolated_count = 0
        for idx, fid in enumerate(valid_id_order):
            neighbor_count = len([nid for nid in neighbors.get(fid, []) if nid in valid_ids])
            if neighbor_count == 0:
                isolated_count += 1
            results_map[fid] = (i_vals[idx], z_scores[idx], p_values[idx], quadrants[idx], neighbor_count)
        if isolated_count:
            feedback.pushWarning(
                f"{isolated_count} feature(s) had no valid neighbors. Consider a larger distance band or K value."
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
                i_val, z, p, quad, neighbor_count = results_map[fid]
                out_feat.setAttribute("bilisa_i", float(i_val))
                out_feat.setAttribute("bilisa_z", float(z))
                out_feat.setAttribute("bilisa_p", float(p))
                out_feat.setAttribute("quadrant", str(quad))
                out_feat.setAttribute("bilisa_nb", int(neighbor_count))
            else:
                out_feat.setAttribute("bilisa_i", None)
                out_feat.setAttribute("bilisa_z", None)
                out_feat.setAttribute("bilisa_p", None)
                out_feat.setAttribute("quadrant", "Not Significant")
                out_feat.setAttribute("bilisa_nb", None)

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
        apply_output_metadata(
            layer,
            "PlanX GeoStats Bivariate Local Moran cluster and outlier output",
            {
                "bilisa_i": "Bivariate Local Moran's I statistic",
                "bilisa_z": "Bivariate Local Moran z-score",
                "bilisa_p": "Bivariate Local Moran p-value",
                "quadrant": "Bivariate LISA class: HH, LL, HL, LH, or Not Significant",
                "bilisa_nb": "Valid neighbors used for the local statistic",
            },
            self.displayName(),
        )
        categories = []
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
