# -*- coding: utf-8 -*-
"""Processing provider registration for PlanX-GeoStats."""
from __future__ import annotations

import os

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .algorithms.alg_dependency_installer import DependencyInstallerAlgorithm
from .algorithms.alg_getis_ord import GetisOrdAlgorithm
from .algorithms.alg_mean_center import MeanCenterAlgorithm
from .algorithms.alg_sde import SDEAlgorithm
from .algorithms.alg_local_moran import LocalMoranAlgorithm
from .algorithms.alg_spatial_regression import SpatialRegressionAlgorithm
from .algorithms.alg_global_moran import GlobalMoranAlgorithm
from .algorithms.alg_average_nearest_neighbor import AverageNearestNeighborAlgorithm
from .algorithms.alg_standard_distance import StandardDistanceAlgorithm
from .algorithms.alg_gwr import GWRAlgorithm
from .algorithms.alg_median_center import MedianCenterAlgorithm
from .algorithms.alg_general_g import GeneralGAlgorithm
from .algorithms.alg_similarity_search import SimilaritySearchAlgorithm
from .algorithms.alg_calculate_distance_band import CalculateDistanceBandAlgorithm
from .algorithms.alg_multivariate_clustering import MultivariateClusteringAlgorithm
from .algorithms.alg_export_attributes import ExportAttributesAlgorithm


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
        self.addAlgorithm(MeanCenterAlgorithm())
        self.addAlgorithm(SDEAlgorithm())
        self.addAlgorithm(LocalMoranAlgorithm())
        self.addAlgorithm(SpatialRegressionAlgorithm())
        self.addAlgorithm(GlobalMoranAlgorithm())
        self.addAlgorithm(AverageNearestNeighborAlgorithm())
        self.addAlgorithm(StandardDistanceAlgorithm())
        self.addAlgorithm(GWRAlgorithm())
        self.addAlgorithm(MedianCenterAlgorithm())
        self.addAlgorithm(GeneralGAlgorithm())
        self.addAlgorithm(SimilaritySearchAlgorithm())
        self.addAlgorithm(CalculateDistanceBandAlgorithm())
        self.addAlgorithm(MultivariateClusteringAlgorithm())
        self.addAlgorithm(ExportAttributesAlgorithm())
        self.addAlgorithm(DependencyInstallerAlgorithm())

