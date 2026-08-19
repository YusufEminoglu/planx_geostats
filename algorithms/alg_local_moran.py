# -*- coding: utf-8 -*-
"""Anselin Local Moran's I (LISA) Processing Algorithm."""
from __future__ import annotations

import logging
import numpy as np

from qgis.PyQt.QtCore import QVariant
from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsProject,
    QgsFeature,
    QgsField,
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
from ..core.layer_metadata import apply_output_metadata
from ..core.local_pattern_audit import local_moran_class_summary
from ..core.symbology import apply_renderer, lisa_quadrant_renderer

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class LocalMoranAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
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

    def icon(self):
        return algorithm_icon("local_moran_lisa")

    def createInstance(self):
        return LocalMoranAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Computes Anselin's Local Moran's I for every feature, classifying "
            "each into a spatial cluster or outlier quadrant using the analytical "
            "randomization variance (Anselin, 1995) - not permutation. Output "
            "fields: lisa_i (local I), lisa_z, lisa_p, lisa_nbrs (valid neighbor "
            "count), and quadrant, assigned only when lisa_p < 0.05:\n"
            "- HH (High-High): a high value surrounded by high neighbors (part of a hot cluster)\n"
            "- LL (Low-Low): a low value surrounded by low neighbors (part of a cold cluster)\n"
            "- HL (High-Low): a high value surrounded by low neighbors (spatial outlier)\n"
            "- LH (Low-High): a low value surrounded by high neighbors (spatial outlier)\n"
            "- Not Significant: p >= 0.05, or the feature had no valid neighbors\n\n"
            "Each feature is tested independently at p < 0.05 with no "
            "multiple-testing correction applied - across hundreds of features, "
            "roughly 5% will cross that threshold by chance alone even under "
            "complete spatial randomness. Treat an isolated significant cell with "
            "suspicion unless it forms a spatially coherent group with its "
            "neighbors; a single significant cell surrounded by 'Not Significant' "
            "cells is a common false-positive pattern. Run Global Moran's I first "
            "to confirm a genuine global signal exists before mapping local "
            "quadrants."
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
            if val is None or val == NULL or str(val) == 'NULL':
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
        class_summary = local_moran_class_summary(quadrants)
        feedback.pushInfo(class_summary["message"])

        if feedback.isCanceled():
            return {}

        # Prepare output fields
        out_fields = source.fields()
        out_fields.append(QgsField("lisa_i", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("lisa_z", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("lisa_p", QVariant.Double, len=10, prec=6))
        out_fields.append(QgsField("quadrant", QVariant.String, len=20))
        out_fields.append(QgsField("lisa_nbrs", QVariant.Int))

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
                f"{isolated_count} feature(s) had no valid neighbors. Review lisa_nbrs and consider a larger distance band or K value."
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
                out_feat.setAttribute("lisa_i", float(i_val))
                out_feat.setAttribute("lisa_z", float(z))
                out_feat.setAttribute("lisa_p", float(p))
                out_feat.setAttribute("quadrant", str(quad))
                out_feat.setAttribute("lisa_nbrs", int(neighbor_count))
            else:
                out_feat.setAttribute("lisa_i", None)
                out_feat.setAttribute("lisa_z", None)
                out_feat.setAttribute("lisa_p", None)
                out_feat.setAttribute("quadrant", "Not Significant")
                out_feat.setAttribute("lisa_nbrs", None)

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
            "PlanX GeoStats Local Moran cluster and outlier output",
            {
                "lisa_i": "Anselin Local Moran's I statistic",
                "lisa_z": "Local Moran z-score",
                "lisa_p": "Local Moran p-value",
                "quadrant": "LISA class: HH, LL, HL, LH, or Not Significant",
                "lisa_nbrs": "Valid neighbors used for the local statistic",
            },
            self.displayName(),
        )
        apply_renderer(layer, lisa_quadrant_renderer(layer.geometryType(), "quadrant"))

        return {}
