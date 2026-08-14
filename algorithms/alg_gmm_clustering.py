# -*- coding: utf-8 -*-
"""Gaussian Mixture Model Clustering Processing Algorithm."""
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
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtCore import QVariant

from ..core.layer_metadata import apply_output_metadata
from ..core.ml_engines import fit_gmm
from ..dependencies import optional_dependency_error

from ._icons import algorithm_icon

logger = logging.getLogger("PlanX GeoStats Lab")


class GMMClusteringAlgorithm(HelpUrlMixin, QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    N_COMPONENTS = "N_COMPONENTS"
    COVARIANCE_TYPE = "COVARIANCE_TYPE"
    OUTPUT = "OUTPUT"

    COVARIANCE_TYPES = ["Full (most flexible)", "Tied", "Diagonal", "Spherical (most constrained)"]
    COVARIANCE_KEYS = ["full", "tied", "diag", "spherical"]

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "gmm_clustering"

    def displayName(self) -> str:
        return "Gaussian Mixture Model Clustering"

    def group(self) -> str:
        return "06 | Machine Learning and Explainable AI"

    def groupId(self) -> str:
        return "planx_ml_xai"

    def icon(self):
        return algorithm_icon("gmm_clustering")

    def createInstance(self):
        return GMMClusteringAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Groups features into K clusters by fitting K overlapping "
            "Gaussian (elliptical) distributions to the Z-score standardized "
            "attribute space via expectation-maximization, then assigning each "
            "point to its highest-probability component. Unlike Multivariate "
            "Clustering (K-Means), which assumes roughly spherical, "
            "similarly-sized clusters, GMM allows elongated, differently-"
            "sized, and differently-oriented clusters, and produces a soft "
            "probability of membership in every cluster rather than a single "
            "hard assignment.\n\n"
            "Output: the input layer plus cluster_id (the highest-probability "
            "component) and one gmm_p0, gmm_p1, ... column per component with "
            "that point's membership probability in each - a point with "
            "probability spread across two components (e.g. 0.55 / 0.45) sits "
            "near the boundary between them, unlike K-Means where every point "
            "looks equally 'certain'.\n\n"
            "Covariance type controls how much shape flexibility each "
            "component gets: Full lets every cluster have its own arbitrary "
            "elliptical shape (most flexible, needs the most data per "
            "cluster to fit reliably); Spherical constrains every cluster to "
            "a circular shape (closest to K-Means, most stable on small "
            "datasets). Start with Full; switch to a more constrained option "
            "if the fit does not converge or looks unstable with few records.\n\n"
            "BIC and AIC (printed to the Processing log, lower is better) let "
            "you compare different numbers of components: run this tool at "
            "several K values and pick the one with the lowest BIC rather "
            "than assuming more components is always better."
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
                self.N_COMPONENTS, "Number of components (K)", type=QgsProcessingParameterNumber.Integer,
                defaultValue=3, minValue=2, maxValue=30,
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.COVARIANCE_TYPE, "Covariance type", options=self.COVARIANCE_TYPES, defaultValue=0,
            )
        )
        self.addParameter(QgsProcessingParameterFeatureSink(self.OUTPUT, "Output GMM clustered layer"))

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        fields_list = self.parameterAsFields(parameters, self.FIELDS, context)
        if not fields_list:
            raise QgsProcessingException("At least one analysis field must be selected.")
        n_components = self.parameterAsInt(parameters, self.N_COMPONENTS, context)
        covariance_type = self.COVARIANCE_KEYS[self.parameterAsEnum(parameters, self.COVARIANCE_TYPE, context)]

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

        if len(fids) < n_components:
            raise QgsProcessingException(f"Insufficient complete records ({len(fids)}) for {n_components} components.")

        x = np.array(rows, dtype=float)
        feedback.pushInfo(f"Fitting Gaussian Mixture (K={n_components}, covariance={covariance_type})...")
        try:
            results = fit_gmm(x, n_components=n_components, covariance_type=covariance_type)
        except ImportError as exc:
            raise QgsProcessingException(optional_dependency_error("Gaussian Mixture Model Clustering", ["scikit-learn"], exc))

        feedback.pushInfo(f"Converged={results['converged']}, BIC={results['bic']:.2f}, AIC={results['aic']:.2f}")
        if not results["converged"]:
            feedback.pushWarning("Expectation-maximization did not converge; try a more constrained covariance type or fewer components.")
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric analysis values.")

        out_fields = source.fields()
        out_fields.append(QgsField("cluster_id", QVariant.Int))
        proba_field_names = [f"gmm_p{i}" for i in range(n_components)]
        for field_name in proba_field_names:
            out_fields.append(QgsField(field_name, QVariant.Double, len=10, prec=4))

        sink, dest_id = self.parameterAsSink(parameters, self.OUTPUT, context, out_fields, source.wkbType(), source.sourceCrs())
        self.out_layer_id = dest_id
        result_map = {fid: idx for idx, fid in enumerate(fids)}
        for current, feature in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break
            out_feature = QgsFeature(feature)
            out_feature.setFields(out_fields)
            fid = feature.id()
            if fid in result_map:
                row_idx = result_map[fid]
                out_feature.setAttribute("cluster_id", int(results["labels"][row_idx]))
                for field_name, prob in zip(proba_field_names, results["proba"][row_idx]):
                    out_feature.setAttribute(field_name, float(prob))
            else:
                out_feature.setAttribute("cluster_id", None)
                for field_name in proba_field_names:
                    out_feature.setAttribute(field_name, None)
            sink.addFeature(out_feature, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        layer = context.getMapLayer(self.out_layer_id) if self.out_layer_id else None
        apply_output_metadata(
            layer,
            "PlanX GeoStats Gaussian Mixture clustering output",
            {"cluster_id": "Highest-probability mixture component index for this record"},
            "gmm_clustering",
        )
        return {}
