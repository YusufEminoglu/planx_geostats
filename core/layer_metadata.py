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
        layer.setCustomProperty("planx_geostats:field_count", len(field_descriptions))
    except (AttributeError, RuntimeError, TypeError):
        return

    try:
        fields = layer.fields()
    except (AttributeError, RuntimeError, TypeError):
        fields = None
    for field_name, description in field_descriptions.items():
        try:
            layer.setCustomProperty(f"planx_geostats:field:{field_name}", description)
        except (AttributeError, RuntimeError, TypeError):
            pass
        if fields is None:
            continue
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
