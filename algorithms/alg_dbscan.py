# -*- coding: utf-8 -*-
"""DBSCAN Density-Based Clustering Processing Algorithm."""
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
from ..core.ml_engines import fit_dbscan
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class DBSCANAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    EPS = "EPS"
    MIN_SAMPLES = "MIN_SAMPLES"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "dbscan_clustering"

    def displayName(self) -> str:
        return "DBSCAN Density-Based Clustering"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("dbscan_clustering")

    def createInstance(self):
        return DBSCANAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Groups features into density-based clusters: a point belongs to a "
            "cluster if at least Min samples other points fall within Epsilon "
            "distance of it in the (Z-score standardized) attribute space, "
            "clusters grow by chaining such neighborhoods together, and any "
            "point that does not qualify is labeled noise (cluster_id = -1) "
            "rather than forced into the nearest cluster.\n\n"
            "Unlike Multivariate Clustering (K-Means), DBSCAN needs no "
            "predetermined number of clusters, finds clusters of arbitrary "
            "shape (not just roughly convex/similarly sized ones), and "
            "explicitly separates outliers from cluster members instead of "
            "assigning every point to the nearest centroid regardless of fit.\n\n"
            "Output: the input layer plus cluster_id (-1 = noise, 0/1/2... = "
            "cluster membership).\n\n"
            "Epsilon is the hardest parameter to set - it is in standardized "
            "attribute-space units, not the fields' original units, so start "
            "around 0.5-1.0 and adjust: too small produces almost all noise, "
            "too large merges everything into one cluster. Watch the noise "
            "count in the Processing log; a run with more than half the "
            "records labeled noise usually means Epsilon is too small or "
            "Min samples is too high for this dataset's density."
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
                self.EPS, "Epsilon (standardized-space neighborhood radius)", type=QgsProcessingParameterNumber.Double,
                defaultValue=0.75, minValue=0.01, maxValue=20.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_SAMPLES, "Min samples per neighborhood", type=QgsProcessingParameterNumber.Integer,
                defaultValue=5, minValue=2, maxValue=200,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output DBSCAN clustered layer"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        fields_list = self.parameterAsFields(parameters, self.FIELDS, context)
        if not fields_list:
            raise QgsProcessingException("At least one analysis field must be selected.")
        eps = self.parameterAsDouble(parameters, self.EPS, context)
        min_samples = self.parameterAsInt(parameters, self.MIN_SAMPLES, context)

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

        if len(fids) < min_samples:
            raise QgsProcessingException(f"Insufficient complete records ({len(fids)}) for Min samples={min_samples}.")

        x = np.array(rows, dtype=float)
        feedback.pushInfo(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
        try:
            results = fit_dbscan(x, eps=eps, min_samples=min_samples)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("DBSCAN Density-Based Clustering", ["scikit-learn"], exc))

        feedback.pushInfo(f"Found {results['n_clusters']} cluster(s); {results['n_noise']} noise point(s) of {len(fids)}.")
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric analysis values.")

        out_fields = source.fields()
        out_fields.append(QgsField("cluster_id", QVariant.Int))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        label_map = {fid: int(label) for fid, label in zip(fids, results["labels"])}
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            out_feature.setAttribute("cluster_id", label_map.get(feature.id()))
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats DBSCAN clustering output",
            {"cluster_id": "DBSCAN cluster index, or -1 for noise/outlier points not assigned to any cluster"},
            "dbscan_clustering",
        )
        return {}
