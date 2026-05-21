# -*- coding: utf-8 -*-
"""Similarity Search Processing Algorithm."""
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
    QgsRendererRange,
    QgsGraduatedSymbolRenderer,
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterField,
    QgsProcessingParameterExpression,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsFeatureSink
)

from ..core.stats_engines import calculate_similarity_search

from ._icons import algorithm_icon


logger = logging.getLogger("PlanX GeoStats Lab")


class SimilaritySearchAlgorithm(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    FIELDS = "FIELDS"
    TARGET_EXPRESSION = "TARGET_EXPRESSION"
    METRIC = "METRIC"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()
        self.out_layer_id = None

    def name(self) -> str:
        return "similarity_search"

    def displayName(self) -> str:
        return "Similarity Search"

    def group(self) -> str:
        return "03 | Hot Spots and Spatial Outliers"

    def groupId(self) -> str:
        return "planx_hotspots_outliers"

    def icon(self):
        return algorithm_icon("similarity_search")

    def createInstance(self):
        return SimilaritySearchAlgorithm()

    def shortHelpString(self) -> str:
        return (
            "Similarity Search finds candidate features that are most similar "
            "to one or more target features, based on a profile of selected numeric fields.\n\n"
            "Target features are identified via a QGIS expression. Attributes are standardized "
            "using Z-scores, and Manhattan or Euclidean distances are calculated between candidates "
            "and the target profile. Output contains rankings (`sim_rank`) and distance scores (`sim_index`)."
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
                "Fields for similarity profile (numeric only)",
                parentLayerParameterName=self.INPUT,
                type=QgsProcessingParameterField.Numeric,
                allowMultiple=True
            )
        )
        self.addParameter(
            QgsProcessingParameterExpression(
                self.TARGET_EXPRESSION,
                "Expression to select target features",
                parentLayerParameterName=self.INPUT,
                defaultValue=""
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                self.METRIC,
                "Distance metric",
                options=["Euclidean", "Manhattan"],
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Output ranked similarity layer"
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Invalid input layer source.")

        compare_fields = self.parameterAsFields(parameters, self.FIELDS, context)
        if not compare_fields:
            raise QgsProcessingException("At least one field must be selected to compute the similarity profile.")

        expr_str = self.parameterAsExpression(parameters, self.TARGET_EXPRESSION, context)
        metric_idx = self.parameterAsEnum(parameters, self.METRIC, context)
        metric = "manhattan" if metric_idx == 1 else "euclidean"

        if not expr_str:
            raise QgsProcessingException("Target selection expression cannot be empty.")

        expr = QgsExpression(expr_str)
        expr_ctx = QgsExpressionContext()
        expr_ctx.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(source))

        if expr.hasParserError():
            raise QgsProcessingException(f"Parser error in expression: {expr.parserErrorString()}")

        # Extract features and profiles
        fids = []
        attribute_matrix = []
        is_target_list = []
        skipped = 0

        field_idxs = [source.fields().lookupField(name) for name in compare_fields]
        for name, idx in zip(compare_fields, field_idxs):
            if idx < 0:
                raise QgsProcessingException(f"Selected profile field '{name}' not found.")

        feedback.pushInfo("Extracting attribute profiles and matching targets...")
        total = source.featureCount() or 1
        for idx, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            # Evaluate target expression
            expr_ctx.setFeature(f)
            is_target = bool(expr.evaluate(expr_ctx))

            # Fetch attribute values
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
            is_target_list.append(is_target)
            feedback.setProgress(int(30 * (idx / total)))

        n = len(fids)
        if n == 0:
            raise QgsProcessingException("No features found with valid numeric attributes for comparison.")
        if skipped:
            feedback.pushInfo(f"Skipped {skipped} feature(s) with missing or non-numeric profile values.")
        field_stds = np.std(np.array(attribute_matrix, dtype=float), axis=0)
        near_constant = [name for name, std in zip(compare_fields, field_stds) if std <= 1e-9]
        if near_constant:
            feedback.pushWarning(
                "Near-constant similarity profile field(s) detected: "
                + ", ".join(near_constant)
                + ". These fields contribute little to standardized profile distance."
            )

        # Identify targets indices
        target_indices = [i for i, target in enumerate(is_target_list) if target]
        if not target_indices:
            raise QgsProcessingException("No features matched the target selection expression.")

        feedback.pushInfo(f"Found {len(target_indices)} target features and {n - len(target_indices)} candidates.")

        full_data = np.array(attribute_matrix)

        feedback.pushInfo("Computing standardized similarity scores...")
        scores = calculate_similarity_search(full_data, target_indices, metric)

        if feedback.isCanceled():
            return {}

        # Rank candidates (targets get Rank 0)
        rankings = np.zeros(n, dtype=int)
        candidate_indices = [i for i in range(n) if not is_target_list[i]]
        
        # Sort candidate indices by score ascending (smaller score = more similar)
        sorted_candidates = sorted(candidate_indices, key=lambda idx: scores[idx])
        
        for rank, idx in enumerate(sorted_candidates, start=1):
            rankings[idx] = rank
        candidate_count = len(candidate_indices)
        similarity_percentiles = np.zeros(n, dtype=float)
        similarity_tiers = ["Target"] * n
        for idx in candidate_indices:
            if candidate_count > 0:
                similarity_percentiles[idx] = 100.0 * (candidate_count - rankings[idx] + 1) / candidate_count
            if rankings[idx] <= 10:
                similarity_tiers[idx] = "Top 10"
            elif similarity_percentiles[idx] >= 75:
                similarity_tiers[idx] = "High"
            elif similarity_percentiles[idx] >= 50:
                similarity_tiers[idx] = "Moderate"
            else:
                similarity_tiers[idx] = "Low"

        # Set up output fields
        out_fields = source.fields()
        out_fields.append(QgsField("is_target", QVariant.Int))
        out_fields.append(QgsField("sim_index", QVariant.Double, len=12, prec=6))
        out_fields.append(QgsField("sim_rank", QVariant.Int))
        out_fields.append(QgsField("sim_pct", QVariant.Double, len=10, prec=3))
        out_fields.append(QgsField("sim_tier", QVariant.String, len=20))

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            source.wkbType(),
            source.sourceCrs()
        )
        self.out_layer_id = dest_id

        # Write ranked features
        results_map = {fids[i]: i for i in range(n)}
        
        feedback.pushInfo("Writing ranked candidates to destination layer...")
        for current, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            out_feat = QgsFeature(f)
            out_feat.setFields(out_fields)

            fid = f.id()
            if fid in results_map:
                idx = results_map[fid]
                out_feat.setAttribute("is_target", 1 if is_target_list[idx] else 0)
                out_feat.setAttribute("sim_index", float(scores[idx]))
                out_feat.setAttribute("sim_rank", int(rankings[idx]))
                out_feat.setAttribute("sim_pct", float(similarity_percentiles[idx]))
                out_feat.setAttribute("sim_tier", similarity_tiers[idx])
            else:
                out_feat.setAttribute("is_target", 0)
                out_feat.setAttribute("sim_index", None)
                out_feat.setAttribute("sim_rank", None)
                out_feat.setAttribute("sim_pct", None)
                out_feat.setAttribute("sim_tier", None)

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)
            feedback.setProgress(int(30 + 70 * (current / total)))

        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        if self.out_layer_id is None:
            return {}

        layer = QgsProject.instance().mapLayer(self.out_layer_id)
        if not layer:
            return {}

        feedback.pushInfo("Applying Similarity Search graduated styling...")

        # Graduated symbol renderer on sim_rank (Top similarity)
        # Ranks: Target is 0, top candidates are 1..10, etc.
        ranges = []
        
        # Color palette: Target is dark grey, candidates scale from high similarity (green) to low (red)
        symbol_target = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol_target.setColor(QColor("#2d3748"))  # Target color
        r_target = QgsRendererRange(0, 0, symbol_target, "Target Feature")
        ranges.append(r_target)

        # Graduated ranges for candidates
        cand_defs = [
            (1, 10, '#2b9348', 'Top 10 Most Similar'),
            (11, 50, '#55a630', 'Very High Similarity (11-50)'),
            (51, 200, '#aacc00', 'Moderate Similarity (51-200)'),
            (201, 1000000, '#ffb703', 'Lower Similarity (>200)')
        ]

        for min_v, max_v, color_hex, label in cand_defs:
            symbol = QgsSymbol.defaultSymbol(layer.geometryType())
            symbol.setColor(QColor(color_hex))
            symbol.setOpacity(0.85)

            if symbol.symbolLayerCount() > 0:
                sl = symbol.symbolLayer(0)
                if hasattr(sl, 'setStrokeColor'):
                    sl.setStrokeColor(QColor('#ffffff'))
                if hasattr(sl, 'setStrokeWidth'):
                    sl.setStrokeWidth(0.1)

            r_range = QgsRendererRange(min_v, max_v, symbol, label)
            ranges.append(r_range)

        renderer = QgsGraduatedSymbolRenderer('sim_rank', ranges)
        layer.setRenderer(renderer)
        layer.triggerRepaint()

        return {}
