# -*- coding: utf-8 -*-
"""PlanX-GeoStats — Spatial Statistics suite for QGIS.

Registers a QgsProcessingProvider in initGui and removes it in unload.
No custom UI; algorithms appear under the Processing Toolbox.
"""
from __future__ import annotations

from qgis.core import QgsApplication

from .planx_geostats_provider import PlanXGeoStatsProvider


class PlanXGeoStatsPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider: PlanXGeoStatsProvider | None = None

    def initGui(self) -> None:
        self.provider = PlanXGeoStatsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self) -> None:
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
