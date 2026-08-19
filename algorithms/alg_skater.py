# -*- coding: utf-8 -*-
"""SKATER Spatially Constrained Regionalization Processing Algorithm."""
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
    QgsFeatureSink,
)

from ..core.weights import build_weights_matrix
from ..core.advanced_stats_engines import calculate_skater
from ..core.layer_metadata import apply_output_metadata
from ..core.symbology import apply_renderer, categorical_id_renderer

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class SkaterAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    WEIGHT_TYPE = "WEIGHT_TYPE"
    KNN = "KNN"
    DISTANCE_BAND = "DISTANCE_BAND"
    K_CLUSTERS = "K_CLUSTERS"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "skater_regionalization"

    def displayName(self) -> str:
        return "Spatially Constrained Regionalization (SKATER)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("skater_regionalization")

    def createInstance(self):
        return SkaterAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Partitions the study area into K spatially CONTIGUOUS regions "
            "using SKATER (Spatial 'K'luster Analysis by Tree Edge Removal; "
            "Assunção et al., 2006) - the direct fix for the plugin's "
            "existing Multivariate Clustering (K-Means) tool's biggest "
            "limitation: K-Means groups by attribute similarity alone, with "
            "no guarantee that a cluster's members are geographically near "
            "each other. SKATER guarantees every output cluster is a single "
            "connected region.\n\n"
            "How it works: every selected numeric field is Z-score "
            "standardized, a minimum spanning tree (MST) is built over the "
            "spatial contiguity graph with edge weights equal to attribute "
            "dissimilarity between neighbors, then the K-1 tree edges whose "
            "removal most reduces total within-cluster sum of squares (SSD) "
            "are cut in turn - a divisive procedure, the reverse of "
            "hierarchical agglomerative clustering. Output is the original "
            "layer plus a region_id field (0 to K-1) and region_ssd (this "
            "region's own within-cluster sum of squares).\n\n"
            "Requires the spatial contiguity graph to be fully connected - "
            "an isolated feature or a study area with disconnected 'islands' "
            "under the chosen weight type will raise a clear error naming "
            "how many features are unreachable; widen K or the distance band "
            "rather than switching to K-Means to work around it, since that "
            "would silently give up the spatial-contiguity guarantee that is "
            "the whole point of this tool. Ships as a native implementation "
            "(NumPy + Prim's algorithm) with no optional-library dependency. "
            "Total SSD across all regions is printed to the Processing log - "
            "same 'run at a few K values and look for the elbow' guidance as "
            "K-Means applies here too."
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
                self.FIELDS,
                "Analysis fields (numeric only)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.WEIGHT_TYPE,
                "Spatial contiguity / relationship type",
                options=["Queen contiguity", "Rook contiguity", "K-Nearest Neighbors (KNN)", "Distance Band"],
                defaultValue=0,
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
        self.addParameter(
            QgsProcessingParameterNumber(
                self.K_CLUSTERS,
                "Number of regions (K)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=2,
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output regionalized layer",
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        compare_fields = self.parameterAsFields(parameters, self.FIELDS, context)
        if not compare_fields:
            raise QgsProcessingException("At least one analysis field must be selected.")

        weight_type_idx = self.parameterAsEnum(parameters, self.WEIGHT_TYPE, context)
        weight_types = ["queen", "rook", "knn", "distance"]
        weight_type = weight_types[weight_type_idx]
        k_neighbors_param = self.parameterAsInt(parameters, self.KNN, context)
        distance_band = self.parameterAsDouble(parameters, self.DISTANCE_BAND, context)
        k_clusters = self.parameterAsInt(parameters, self.K_CLUSTERS, context)

        field_idxs = [source.fields().lookupField(name) for name in compare_fields]
        for name, idx in zip(compare_fields, field_idxs):
            if idx < 0:
                raise QgsProcessingException(f"Selected analysis field '{name}' not found.")

        feedback.pushInfo("Generating spatial contiguity graph...")
        neighbors, _weights, id_order, _ = build_weights_matrix(
            source, weight_type, k_neighbors=k_neighbors_param, distance_band=distance_band, feedback=feedback
        )
        if feedback.isCanceled():
            return {}

        feedback.pushInfo("Extracting attributes for regionalization...")
        attr_by_fid = {}
        skipped = 0
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            has_null = False
            vals = []
            for f_idx in field_idxs:
                val = f.attribute(f_idx)
                if val is None or val == NULL or str(val) == "NULL":
                    has_null = True
                    break
                try:
                    vals.append(float(val))
                except (ValueError, TypeError):
                    has_null = True
                    break
            if has_null:
                skipped += 1
                continue
            attr_by_fid[f.id()] = vals
            feedback.setProgress(int(30 * (idx / total)))

        valid_id_order = [fid for fid in id_order if fid in attr_by_fid]
        n = len(valid_id_order)
        if n < k_clusters:
            raise QgsProcessingException(
                f"Insufficient features ({n}) with valid attributes to form {k_clusters} regions."
            )
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric analysis values.")

        data = np.array([attr_by_fid[fid] for fid in valid_id_order])
        field_stds = np.std(data, axis=0)
        near_constant = [name for name, std in zip(compare_fields, field_stds) if std <= 1e-9]
        if near_constant:
            feedback.pushWarning(
                "Near-constant analysis field(s) detected: "
                + ", ".join(near_constant)
                + ". These fields contribute little to regional separation."
            )

        feedback.pushInfo(f"Building minimum spanning tree and cutting {k_clusters - 1} edge(s)...")
        labels, total_ssd = calculate_skater(data, neighbors, valid_id_order, k_clusters)
        feedback.pushInfo(f"Total SSD (Sum of Squared Deviations across all regions): {total_ssd:.4f}")

        region_sizes = np.array([int(np.sum(labels == region)) for region in range(k_clusters)])
        feedback.pushInfo(
            "Region size diagnostics: " + ", ".join(f"Region {idx}: {size}" for idx, size in enumerate(region_sizes))
        )

        z_data = (data - np.mean(data, axis=0)) / np.where(field_stds == 0.0, 1.0, field_stds)
        region_ssd = np.zeros(k_clusters)
        for region in range(k_clusters):
            mask = labels == region
            if np.any(mask):
                centroid = np.mean(z_data[mask], axis=0)
                region_ssd[region] = np.sum((z_data[mask] - centroid) ** 2)

        if feedback.isCanceled():
            return {}

        out_fields = source.fields()
        out_fields.append(QgsField("region_id", QVariant.Int))
        out_fields.append(QgsField("region_size", QVariant.Int))
        out_fields.append(QgsField("region_ssd", QVariant.Double, len=12, prec=6))

        (sink, dest_id) = self.parameterAsSink(
            parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs()
        )
        self.out_layer_id = dest_id

        results_map = {
            valid_id_order[i]: (labels[i], region_sizes[labels[i]], region_ssd[labels[i]])
            for i in range(n)
        }

        feedback.pushInfo("Writing regionalized features to destination layer...")
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                region, size, ssd = results_map[fid]
                out_feat.setAttribute("region_id", int(region))
                out_feat.setAttribute("region_size", int(size))
                out_feat.setAttribute("region_ssd", float(ssd))
            else:
                out_feat.setAttribute("region_id", None)
                out_feat.setAttribute("region_size", None)
                out_feat.setAttribute("region_ssd", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        region_idx = layer.fields().lookupField("region_id")
        if region_idx < 0:
            return {}

        unique_regions = sorted(v for v in layer.uniqueValues(region_idx) if v is not None and v != NULL)
        if not unique_regions:
            return {}

        feedback.pushInfo("Applying SKATER regionalization styling...")
        apply_output_metadata(
            layer,
            "PlanX GeoStats SKATER spatially constrained regionalization output",
            {
                "region_id": "SKATER region identifier (spatially contiguous by construction)",
                "region_size": "Number of complete records assigned to this region",
                "region_ssd": "Standardized within-region sum of squared deviations",
            },
            self.displayName(),
        )

        apply_renderer(layer, categorical_id_renderer(layer, layer.geometryType(), "region_id"))

        return {}
