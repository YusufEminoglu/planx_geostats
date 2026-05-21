# -*- coding: utf-8 -*-
"""Processing provider registration for PlanX GeoStats Lab."""
from __future__ import annotations

import os

from qgis.PyQt.QtGui import QIcon
from qgis.core import QgsProcessingProvider

from .algorithms.alg_getis_ord import GetisOrdAlgorithm
from .algorithms.alg_mean_center import MeanCenterAlgorithm
from .algorithms.alg_sde import SDEAlgorithm
from .algorithms.alg_local_moran import LocalMoranAlgorithm
from .algorithms.alg_bivariate_lee_l import BivariateLeeLAlgorithm
from .algorithms.alg_spatial_regression import SpatialRegressionAlgorithm
from .algorithms.alg_spatial_autoregression import SpatialAutoregressionAlgorithm
from .algorithms.alg_spatial_error_regression import SpatialErrorRegressionAlgorithm
from .algorithms.alg_global_moran import GlobalMoranAlgorithm
from .algorithms.alg_incremental_autocorrelation import IncrementalAutocorrelationAlgorithm
from .algorithms.alg_ripleys_k import RipleysKFunctionAlgorithm
from .algorithms.alg_average_nearest_neighbor import AverageNearestNeighborAlgorithm
from .algorithms.alg_standard_distance import StandardDistanceAlgorithm
from .algorithms.alg_gwr import GWRAlgorithm
from .algorithms.alg_mgwr import MGWRAlgorithm
from .algorithms.alg_exploratory_regression import ExploratoryRegressionAlgorithm
from .algorithms.alg_generalized_linear_regression import GeneralizedLinearRegressionAlgorithm
from .algorithms.alg_model_comparison import ModelComparisonAlgorithm
from .algorithms.alg_median_center import MedianCenterAlgorithm
from .algorithms.alg_central_feature import CentralFeatureAlgorithm
from .algorithms.alg_general_g import GeneralGAlgorithm
from .algorithms.alg_similarity_search import SimilaritySearchAlgorithm
from .algorithms.alg_calculate_distance_band import CalculateDistanceBandAlgorithm
from .algorithms.alg_multivariate_clustering import MultivariateClusteringAlgorithm
from .algorithms.alg_export_attributes import ExportAttributesAlgorithm
from .algorithms.alg_linear_directional_mean import LinearDirectionalMeanAlgorithm
from .algorithms.alg_sensitivity_test import SensitivityTestAlgorithm
from .algorithms.alg_library_status import GeoStatsLibraryStatusAlgorithm
from .algorithms.alg_install_libraries import InstallGeoStatsLibrariesAlgorithm
from .algorithms.alg_sample_data_guide import SampleDataGuideAlgorithm
from .algorithms.alg_data_readiness_audit import DataReadinessAuditAlgorithm
from .algorithms.alg_workflow_advisor import GeoStatsWorkflowAdvisorAlgorithm


class PlanXGeoStatsProvider(QgsProcessingProvider):
    PROVIDER_ID = "planx_geostats"
    PROVIDER_NAME = "PlanX GeoStats Lab"

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
        # 00 | Setup and Diagnostics
        self.addAlgorithm(GeoStatsLibraryStatusAlgorithm())
        self.addAlgorithm(InstallGeoStatsLibrariesAlgorithm())
        self.addAlgorithm(SampleDataGuideAlgorithm())
        self.addAlgorithm(DataReadinessAuditAlgorithm())
        self.addAlgorithm(GeoStatsWorkflowAdvisorAlgorithm())

        # 01 | Data Preparation and Neighborhoods
        self.addAlgorithm(CalculateDistanceBandAlgorithm())
        self.addAlgorithm(ExportAttributesAlgorithm())

        # 02 | Urban Pattern Scan
        self.addAlgorithm(GlobalMoranAlgorithm())
        self.addAlgorithm(GeneralGAlgorithm())
        self.addAlgorithm(IncrementalAutocorrelationAlgorithm())
        self.addAlgorithm(RipleysKFunctionAlgorithm())
        self.addAlgorithm(AverageNearestNeighborAlgorithm())

        # 03 | Hot Spots and Spatial Outliers
        self.addAlgorithm(GetisOrdAlgorithm())
        self.addAlgorithm(LocalMoranAlgorithm())
        self.addAlgorithm(BivariateLeeLAlgorithm())
        self.addAlgorithm(MultivariateClusteringAlgorithm())
        self.addAlgorithm(SimilaritySearchAlgorithm())

        # 04 | Centers, Direction and Dispersion
        self.addAlgorithm(MeanCenterAlgorithm())
        self.addAlgorithm(CentralFeatureAlgorithm())
        self.addAlgorithm(MedianCenterAlgorithm())
        self.addAlgorithm(StandardDistanceAlgorithm())
        self.addAlgorithm(SDEAlgorithm())
        self.addAlgorithm(LinearDirectionalMeanAlgorithm())

        # 05 | Models and Scenarios
        self.addAlgorithm(SpatialRegressionAlgorithm())
        self.addAlgorithm(GeneralizedLinearRegressionAlgorithm())
        self.addAlgorithm(SpatialAutoregressionAlgorithm())
        self.addAlgorithm(SpatialErrorRegressionAlgorithm())
        self.addAlgorithm(ExploratoryRegressionAlgorithm())
        self.addAlgorithm(GWRAlgorithm())
        self.addAlgorithm(MGWRAlgorithm())
        self.addAlgorithm(SensitivityTestAlgorithm())
        self.addAlgorithm(ModelComparisonAlgorithm())
