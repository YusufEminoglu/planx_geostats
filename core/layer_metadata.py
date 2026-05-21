# -*- coding: utf-8 -*-
"""Output-layer metadata helpers for PlanX GeoStats Lab algorithms."""
from __future__ import annotations


def apply_output_metadata(layer, title: str, field_descriptions: dict[str, str], source_algorithm: str) -> None:
    """Attach lightweight aliases/custom properties to an output layer when QGIS permits it."""
    if layer is None:
        return
    try:
        layer.setCustomProperty("planx_geostats:algorithm", source_algorithm)
        layer.setCustomProperty("planx_geostats:title", title)
    except (AttributeError, RuntimeError, TypeError):
        return

    fields = layer.fields()
    for field_name, description in field_descriptions.items():
        try:
            idx = fields.lookupField(field_name)
        except (AttributeError, RuntimeError, TypeError):
            idx = -1
        if idx < 0:
            continue
        try:
            layer.setFieldAlias(idx, description)
        except (AttributeError, RuntimeError, TypeError):
            pass
