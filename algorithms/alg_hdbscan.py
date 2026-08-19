# -*- coding: utf-8 -*-
"""HDBSCAN Hierarchical Density-Based Clustering Processing Algorithm."""
from __future__ import annotations

import logging

import numpy as np

from ._mixins import HelpUrlMixin
from qgis.core import (
    NULL,
    QgsFeature,
    QgsFeatureSink,
    QgsField,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.symbology import apply_renderer, categorical_id_renderer
from ..core.ml_engines import fit_hdbscan
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class HDBSCANAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    MIN_CLUSTER_SIZE = "MIN_CLUSTER_SIZE"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "hdbscan_clustering"

    def displayName(self) -> str:
        return "HDBSCAN Hierarchical Density Clustering"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("hdbscan_clustering")

    def createInstance(self):
        return HDBSCANAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Groups features into density-based clusters like DBSCAN, but "
            "builds a hierarchy of clusters at every density level and "
            "extracts the most stable ones automatically, instead of using one "
            "fixed Epsilon radius everywhere. This handles clusters of "
            "different densities in the same layer (a dense old-town core and "
            "a sparser suburban cluster, for example) that a single-Epsilon "
            "DBSCAN run cannot separate correctly. Requires scikit-learn 1.3 "
            "or newer (Setup and Diagnostics > GeoStats Library Status shows "
            "the installed version).\n\n"
            "Output: the input layer plus cluster_id (-1 = noise) and "
            "cluster_prob, the membership strength (0-1) of each point in its "
            "assigned cluster - a point near a cluster's stable core scores "
            "close to 1, a point near its fuzzy boundary scores lower.\n\n"
            "Min cluster size is the main tuning knob and is more intuitive "
            "than DBSCAN's Epsilon: it is a literal minimum number of points "
            "for a group to count as a cluster, in the same units as your "
            "record count. Raise it to merge small clusters into noise or "
            "into larger neighbors; lower it to allow smaller, tighter groups "
            "to register as their own cluster.\n\n"
            "Prefer this over DBSCAN Density-Based Clustering whenever the "
            "study area plausibly mixes dense and sparse sub-areas, which is "
            "common in mixed urban/suburban planning datasets."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(self.INPUT, "Input vector layer", [QgsProcessing.TypeVectorAnyGeometry])
        )
        self.addParameter(
            QgsProcessingParameterField(
                self.FIELDS, "Analysis fields (numeric only)", parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric, allowMultiple=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_CLUSTER_SIZE, "Min cluster size", type=QgsProcessingParameterNumber.Integer,
                defaultValue=5, minValue=2, maxValue=1000,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output HDBSCAN clustered layer"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        fields_list = self.parameterAsFields(parameters, self.FIELDS, context)
        if not fields_list:
            raise QgsProcessingException("At least one analysis field must be selected.")
        min_cluster_size = self.parameterAsInt(parameters, self.MIN_CLUSTER_SIZE, context)

        field_idxs = [source.fields().lookupField(name) for name in fields_list]
        for name, idx in zip(fields_list, field_idxs):
            if idx < 0:
                raise QgsProcessingException(f"Selected analysis field '{name}' not found.")

        fids, rows, skipped = [], [], 0
        total = source.featureCount() or 1
        feedback.pushInfo("Extracting attributes for clustering...")
        for idx, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            values = []
            has_null = False
            for field_idx in field_idxs:
                value = feature.attribute(field_idx)
                if value is None or value == NULL or str(value) == "NULL":
                    has_null = True
                    break
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    has_null = True
                    break
            if has_null:
                skipped += 1
                continue
            fids.append(feature.id())
            rows.append(values)
            feedback.setProgress(int(30 * (idx / total)))

        if len(fids) < min_cluster_size:
            raise QgsProcessingException(f"Insufficient complete records ({len(fids)}) for Min cluster size={min_cluster_size}.")

        x = np.array(rows, dtype=float)
        feedback.pushInfo(f"Running HDBSCAN (min_cluster_size={min_cluster_size})...")
        try:
            results = fit_hdbscan(x, min_cluster_size=min_cluster_size)
        except ImportError as exc:
            raise QgsProcessingException(
                optional_dependency_error(
                    "HDBSCAN Hierarchical Density Clustering (requires scikit-learn >= 1.3)", ["scikit-learn"], exc
                )
            )

        feedback.pushInfo(f"Found {results['n_clusters']} cluster(s); {results['n_noise']} noise point(s) of {len(fids)}.")
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric analysis values.")

        out_fields = source.fields()
        out_fields.append(QgsField("cluster_id", QVariant.Int))
        out_fields.append(QgsField("cluster_prob", QVariant.Double, len=10, prec=4))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        label_map = {fid: (int(label), float(prob)) for fid, label, prob in zip(fids, results["labels"], results["probabilities"])}
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in label_map:
                label, prob = label_map[fid]
                out_feature.setAttribute("cluster_id", label)
                out_feature.setAttribute("cluster_prob", prob)
            else:
                out_feature.setAttribute("cluster_id", None)
                out_feature.setAttribute("cluster_prob", None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats HDBSCAN clustering output",
            {
                "cluster_id": "HDBSCAN cluster index, or -1 for noise/outlier points not assigned to any cluster",
                "cluster_prob": "Membership strength (0-1) in the assigned cluster; lower values are near the cluster's fuzzy boundary",
            },
            "hdbscan_clustering",
        )
        apply_renderer(layer, categorical_id_renderer(layer, layer.geometryType(), "cluster_id", noise_value=-1))
        return {}
