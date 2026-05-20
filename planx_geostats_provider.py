# -*- coding: utf-8 -*-
"""Processing provider registration for PlanX-GeoStats."""
from __future__ import annotations

import os

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .algorithms.alg_dependency_installer import DependencyInstallerAlgorithm
from .algorithms.alg_getis_ord import GetisOrdAlgorithm


class PlanXGeoStatsProvider(QgsProcessingProvider):
    PROVIDER_ID = "planx_geostats"
    PROVIDER_NAME = "PlanX-GeoStats"

    def id(self) -> str:
        return self.PROVIDER_ID

    def name(self) -> str:
        return self.PROVIDER_NAME

    def longName(self) -> str:
        return self.PROVIDER_NAME

    def icon(self) -> QIcon:
        icon_path = os.path.join(os.path.dirname(__file__), "icons", "icon.png")
        return QIcon(icon_path) if os.path.exists(icon_path) else super().icon()

    def loadAlgorithms(self) -> None:
        # Register the algorithms:
        self.addAlgorithm(GetisOrdAlgorithm())
        self.addAlgorithm(DependencyInstallerAlgorithm())
