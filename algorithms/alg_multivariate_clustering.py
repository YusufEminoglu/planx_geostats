# -*- coding: utf-8 -*-
"""Multivariate Clustering Processing Algorithm."""
from __future__ import annotations

import logging
import numpy as np

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsProject,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsSymbol,
    QgsRendererCategory,
    QgsCategorizedSymbolRenderer,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsFeatureSink
)

from ..core.stats_engines import calculate_kmeans

logger = logging.getLogger("PlanX GeoStats Lab")


class MultivariateClusteringAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    K_CLUSTERS = "K_CLUSTERS"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "multivariate_clustering"

    def displayName(self) -> str:
        return "Multivariate Clustering (K-Means)"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def createInstance(self):
        return MultivariateClusteringAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Performs K-Means clustering on feature attribute values to group features "
            "into a specified number of clusters based on multivariate similarity.\n\n"
            "All input attributes are standardized using Z-scores before clustering. "
            "Outputs a new vector layer containing the original attributes plus a new "
            "`cluster_id` column (0 to K-1) representing cluster assignment."
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
                self.FIELDS,
                "Analysis fields (numeric only)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.K_CLUSTERS,
                "Number of clusters (K)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3,
                minValue=2
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output clustered layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        compare_fields = self.parameterAsFields(parameters, self.FIELDS, context)
        if not compare_fields:
            raise QgsProcessingException("At least one analysis field must be selected.")

        k_clusters = self.parameterAsInt(parameters, self.K_CLUSTERS, context)

        # Get field indexes
        field_idxs = [source.fields().lookupField(name) for name in compare_fields]
        for name, idx in zip(compare_fields, field_idxs):
            if idx < 0:
                raise QgsProcessingException(f"Selected analysis field '{name}' not found.")

        # Extract features and attribute matrix
        fids = []
        attribute_matrix = []
        skipped = 0

        feedback.pushInfo("Extracting attributes for clustering...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            has_null = False
            vals = []
            for f_idx in field_idxs:
                val = f.attribute(f_idx)
                if val is None or val == QVariant() or str(val) == 'NULL':
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

            fids.append(f.id())
            attribute_matrix.append(vals)
            feedback.setProgress(int(40 * (idx / total)))

        n = len(fids)
        if n < k_clusters:
            raise QgsProcessingException(
                f"Insufficient features ({n}) with valid attributes to form {k_clusters} clusters."
            )

        data = np.array(attribute_matrix)
        field_stds = np.std(data, axis=0)
        near_constant = [name for name, std in zip(compare_fields, field_stds) if std <= 1e-9]
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric analysis values.")
        if near_constant:
            feedback.pushWarning(
                "Near-constant clustering field(s) detected: "
                + ", ".join(near_constant)
                + ". These fields contribute little to cluster separation."
            )

        feedback.pushInfo(f"Running K-Means algorithm for {k_clusters} clusters...")
        labels, wcss = calculate_kmeans(data, k_clusters)
        feedback.pushInfo(f"WCSS (Within-Cluster Sum of Squares): {wcss:.4f}")
        z_data = (data - np.mean(data, axis=0)) / np.where(field_stds == 0.0, 1.0, field_stds)
        centroids = np.vstack([np.mean(z_data[labels == cluster], axis=0) for cluster in range(k_clusters)])
        cluster_sizes = np.array([int(np.sum(labels == cluster)) for cluster in range(k_clusters)])
        cluster_distances = np.array([
            float(np.linalg.norm(z_data[idx] - centroids[labels[idx]]))
            for idx in range(n)
        ])
        feedback.pushInfo(
            "Cluster size diagnostics: "
            + ", ".join(f"Cluster {idx}: {size}" for idx, size in enumerate(cluster_sizes))
        )

        if feedback.isCanceled():
            return {}

        # Set up output fields
        out_fields = source.fields()
        out_fields.append(QgsField("cluster_id", QVariant.Int))
        out_fields.append(QgsField("clust_size", QVariant.Int))
        out_fields.append(QgsField("clust_dist", QVariant.Double, len=12, prec=6))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        self.out_layer_id = dest_id

        # Write categorized features
        results_map = {
            fids[i]: (labels[i], cluster_sizes[labels[i]], cluster_distances[i])
            for i in range(n)
        }
        
        feedback.pushInfo("Writing clustered features to destination layer...")
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                label, cluster_size, cluster_distance = results_map[fid]
                out_feat.setAttribute("cluster_id", int(label))
                out_feat.setAttribute("clust_size", int(cluster_size))
                out_feat.setAttribute("clust_dist", float(cluster_distance))
            else:
                out_feat.setAttribute("cluster_id", None)
                out_feat.setAttribute("clust_size", None)
                out_feat.setAttribute("clust_dist", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(40 + 60 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        # Fetch unique cluster IDs from the layer to style them
        cluster_idx = layer.fields().lookupField("cluster_id")
        if cluster_idx < 0:
            return {}

        unique_clusters = sorted(list(layer.uniqueValues(cluster_idx)))
        if not unique_clusters:
            return {}

        feedback.pushInfo("Applying Multivariate Clustering categorized styling...")

        # Distinct premium colors for clusters (categorical/qualitative)
        palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        categories = []
        for idx, cluster_val in enumerate(unique_clusters):
            if cluster_val == QVariant() or cluster_val is None:
                continue
            
            c_val = int(cluster_val)
            color_hex = palette[c_val % len(palette)]
            
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)

            if symbol.symbolLayerCount() > 0:
                sl = symbol.symbolLayer(0)
                if hasattr(sl, 'setStrokeColor'):
                    sl.setStrokeColor(QColor('#ffffff'))
                if hasattr(sl, 'setStrokeWidth'):
                    sl.setStrokeWidth(0.15)

            category = QgsRendererCategory(c_val, symbol, f"Cluster {c_val}")
            categories.append(category)

        renderer = QgsCategorizedSymbolRenderer('cluster_id', categories)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {}
