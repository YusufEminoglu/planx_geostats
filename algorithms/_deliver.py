# -*- coding: utf-8 -*-
"""Make ``postProcessAlgorithm`` styling reach the layer QGIS actually loads.

Every GeoStats tool styles its output in ``postProcessAlgorithm`` through
``context.getMapLayer(self.out_layer_id)``. For a temporary (memory) output
that is the layer QGIS adds to the project. For a GeoPackage or Shapefile
output the lookup opens a *separate* copy of the file: the renderer, field
aliases and metadata land on that copy, and the layer QGIS loads afterwards
arrives unstyled (measured on 42 of 43 styled tools, QGIS 3.44 and 4.2).

``@style_loaded_output`` keeps the body unchanged but, for a file output that
QGIS is about to load, runs it from a layer post-processor instead, with a
context view whose ``getMapLayer(out_layer_id)`` returns the loaded layer.
QGIS keeps only a raw pointer to a post-processor, so each one is kept alive
here (never rotated: batch runs load every result at the end).
"""
from __future__ import annotations

import functools

from qgis.core import QgsProcessingLayerPostProcessorInterface

_ALIVE: list = []


class _LoadedLayerContext:
    """Stand-in context: resolves the output id to the loaded layer and
    delegates everything else to the real processing context."""

    def __init__(self, context, dest_id, layer):
        self._context = context
        self._dest_id = dest_id
        self._layer = layer

    def getMapLayer(self, identifier):
        if identifier == self._dest_id:
            return self._layer
        return self._context.getMapLayer(identifier)

    def __getattr__(self, name):
        return getattr(self._context, name)


class _DeferredStyling(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, algorithm, body, dest_id):
        super().__init__()
        self._algorithm = algorithm
        self._body = body
        self._dest_id = dest_id

    def postProcessLayer(self, layer, context, feedback):
        if layer is None:
            return
        try:
            self._body(self._algorithm, _LoadedLayerContext(context, self._dest_id, layer), feedback)
            layer.triggerRepaint()
        except Exception as exc:
            if feedback is not None:
                feedback.reportError(f"Could not style the loaded output: {exc}", False)


def style_loaded_output(body):
    """Decorator for ``postProcessAlgorithm(self, context, feedback)``."""

    @functools.wraps(body)
    def wrapper(self, context, feedback):
        dest_id = getattr(self, "out_layer_id", None)
        store = context.temporaryLayerStore() if hasattr(context, "temporaryLayerStore") else None
        in_store = store is not None and dest_id and store.mapLayer(dest_id) is not None
        if dest_id and not in_store and context.willLoadLayerOnCompletion(dest_id):
            processor = _DeferredStyling(self, body, dest_id)
            _ALIVE.append(processor)
            context.layerToLoadOnCompletionDetails(dest_id).setPostProcessor(processor)
            return {}
        return body(self, context, feedback)

    return wrapper
