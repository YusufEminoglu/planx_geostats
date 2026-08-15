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
from .algorithms.alg_bivariate_lisa import BivariateLISAAlgorithm
from .algorithms.alg_bivariate_lee_l import BivariateLeeLAlgorithm
from .algorithms.alg_spatial_regression import SpatialRegressionAlgorithm
from .algorithms.alg_spatial_autoregression import SpatialAutoregressionAlgorithm
from .algorithms.alg_spatial_error_regression import SpatialErrorRegressionAlgorithm
from .algorithms.alg_global_moran import GlobalMoranAlgorithm
from .algorithms.alg_spatial_gini import SpatialGiniAlgorithm
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
from .algorithms.alg_geary_c import GearyCAlgorithm
from .algorithms.alg_join_count import JoinCountStatisticsAlgorithm
from .algorithms.alg_global_lee_l import GlobalLeesLAlgorithm
from .algorithms.alg_geodetector_q import GeodetectorQAlgorithm
from .algorithms.alg_local_geary_c import LocalGearyCAlgorithm
from .algorithms.alg_colocation_quotient import ColocationQuotientAlgorithm
from .algorithms.alg_skater import SkaterAlgorithm
from .algorithms.alg_lm_diagnostics import LMDiagnosticsAlgorithm
from .algorithms.alg_spatial_durbin import SpatialDurbinAlgorithm
from .algorithms.alg_esf_regression import ESFRegressionAlgorithm
from .algorithms.alg_random_forest_regression import RandomForestRegressionAlgorithm
from .algorithms.alg_random_forest_classification import RandomForestClassificationAlgorithm
from .algorithms.alg_extra_trees_regression import ExtraTreesRegressionAlgorithm
from .algorithms.alg_extra_trees_classification import ExtraTreesClassificationAlgorithm
from .algorithms.alg_svr import SVRAlgorithm
from .algorithms.alg_svc import SVCAlgorithm
from .algorithms.alg_gbm_regression_sklearn import GBMRegressionSklearnAlgorithm
from .algorithms.alg_gbm_regression_xgboost import GBMRegressionXGBoostAlgorithm
from .algorithms.alg_gbm_regression_lightgbm import GBMRegressionLightGBMAlgorithm
from .algorithms.alg_gbm_classification_sklearn import GBMClassificationSklearnAlgorithm
from .algorithms.alg_gbm_classification_xgboost import GBMClassificationXGBoostAlgorithm
from .algorithms.alg_gbm_classification_lightgbm import GBMClassificationLightGBMAlgorithm
from .algorithms.alg_gbm_regression_catboost import GBMRegressionCatBoostAlgorithm
from .algorithms.alg_gbm_classification_catboost import GBMClassificationCatBoostAlgorithm
from .algorithms.alg_mlp_regression import MLPRegressionAlgorithm
from .algorithms.alg_mlp_classification import MLPClassificationAlgorithm
from .algorithms.alg_spatial_cv_evaluator import SpatialCVEvaluatorAlgorithm
from .algorithms.alg_permutation_importance import PermutationImportanceAlgorithm
from .algorithms.alg_partial_dependence import PartialDependenceAlgorithm
from .algorithms.alg_ml_model_comparison import MLModelComparisonAlgorithm
from .algorithms.alg_shap_global_importance import SHAPGlobalImportanceAlgorithm
from .algorithms.alg_shap_spatial_map import SHAPSpatialMapAlgorithm
from .algorithms.alg_shap_local_explanation import SHAPLocalExplanationAlgorithm
from .algorithms.alg_model_residual_autocorrelation import ModelResidualAutocorrelationAlgorithm
from .algorithms.alg_prediction_uncertainty_map import PredictionUncertaintyMapAlgorithm
from .algorithms.alg_dbscan import DBSCANAlgorithm
from .algorithms.alg_hdbscan import HDBSCANAlgorithm
from .algorithms.alg_gmm_clustering import GMMClusteringAlgorithm
from .algorithms.alg_spatial_regime_regression import SpatialRegimeRegressionAlgorithm
from .algorithms.alg_quantile_regression import QuantileRegressionAlgorithm
from .algorithms.alg_gw_summary_stats import GWSummaryStatsAlgorithm
from .algorithms.alg_conformal_prediction import ConformalPredictionAlgorithm
from .algorithms.alg_tabpfn_regression import TabPFNRegressionAlgorithm
from .algorithms.alg_tabpfn_classification import TabPFNClassificationAlgorithm
from .algorithms.alg_dice_counterfactual import DiCECounterfactualAlgorithm
from .algorithms.alg_ebm_regression import EBMRegressionAlgorithm
from .algorithms.alg_ebm_classification import EBMClassificationAlgorithm


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
        self.addAlgorithm(SpatialGiniAlgorithm())
        self.addAlgorithm(GeneralGAlgorithm())
        self.addAlgorithm(IncrementalAutocorrelationAlgorithm())
        self.addAlgorithm(RipleysKFunctionAlgorithm())
        self.addAlgorithm(AverageNearestNeighborAlgorithm())
        self.addAlgorithm(GearyCAlgorithm())
        self.addAlgorithm(JoinCountStatisticsAlgorithm())
        self.addAlgorithm(GlobalLeesLAlgorithm())
        self.addAlgorithm(GeodetectorQAlgorithm())

        # 03 | Hot Spots and Spatial Outliers
        self.addAlgorithm(GetisOrdAlgorithm())
        self.addAlgorithm(LocalMoranAlgorithm())
        self.addAlgorithm(BivariateLISAAlgorithm())
        self.addAlgorithm(BivariateLeeLAlgorithm())
        self.addAlgorithm(MultivariateClusteringAlgorithm())
        self.addAlgorithm(SimilaritySearchAlgorithm())
        self.addAlgorithm(LocalGearyCAlgorithm())
        self.addAlgorithm(ColocationQuotientAlgorithm())
        self.addAlgorithm(SkaterAlgorithm())

        # 04 | Centers, Direction and Dispersion
        self.addAlgorithm(MeanCenterAlgorithm())
        self.addAlgorithm(CentralFeatureAlgorithm())
        self.addAlgorithm(MedianCenterAlgorithm())
        self.addAlgorithm(StandardDistanceAlgorithm())
        self.addAlgorithm(SDEAlgorithm())
        self.addAlgorithm(LinearDirectionalMeanAlgorithm())

        # 05 | Models and Scenarios
        self.addAlgorithm(SpatialRegressionAlgorithm())
        self.addAlgorithm(LMDiagnosticsAlgorithm())
        self.addAlgorithm(GeneralizedLinearRegressionAlgorithm())
        self.addAlgorithm(SpatialAutoregressionAlgorithm())
        self.addAlgorithm(SpatialErrorRegressionAlgorithm())
        self.addAlgorithm(SpatialDurbinAlgorithm())
        self.addAlgorithm(ExploratoryRegressionAlgorithm())
        self.addAlgorithm(GWRAlgorithm())
        self.addAlgorithm(MGWRAlgorithm())
        self.addAlgorithm(ESFRegressionAlgorithm())
        self.addAlgorithm(SpatialRegimeRegressionAlgorithm())
        self.addAlgorithm(QuantileRegressionAlgorithm())
        self.addAlgorithm(GWSummaryStatsAlgorithm())
        self.addAlgorithm(SensitivityTestAlgorithm())
        self.addAlgorithm(ModelComparisonAlgorithm())

        # 06 | Machine Learning and Explainable AI
        self.addAlgorithm(RandomForestRegressionAlgorithm())
        self.addAlgorithm(RandomForestClassificationAlgorithm())
        self.addAlgorithm(ExtraTreesRegressionAlgorithm())
        self.addAlgorithm(ExtraTreesClassificationAlgorithm())
        self.addAlgorithm(SVRAlgorithm())
        self.addAlgorithm(SVCAlgorithm())
        self.addAlgorithm(GBMRegressionSklearnAlgorithm())
        self.addAlgorithm(GBMRegressionXGBoostAlgorithm())
        self.addAlgorithm(GBMRegressionLightGBMAlgorithm())
        self.addAlgorithm(GBMClassificationSklearnAlgorithm())
        self.addAlgorithm(GBMClassificationXGBoostAlgorithm())
        self.addAlgorithm(GBMClassificationLightGBMAlgorithm())
        self.addAlgorithm(GBMRegressionCatBoostAlgorithm())
        self.addAlgorithm(GBMClassificationCatBoostAlgorithm())
        self.addAlgorithm(MLPRegressionAlgorithm())
        self.addAlgorithm(MLPClassificationAlgorithm())
        self.addAlgorithm(SpatialCVEvaluatorAlgorithm())
        self.addAlgorithm(PermutationImportanceAlgorithm())
        self.addAlgorithm(PartialDependenceAlgorithm())
        self.addAlgorithm(MLModelComparisonAlgorithm())
        self.addAlgorithm(SHAPGlobalImportanceAlgorithm())
        self.addAlgorithm(SHAPSpatialMapAlgorithm())
        self.addAlgorithm(SHAPLocalExplanationAlgorithm())
        self.addAlgorithm(DiCECounterfactualAlgorithm())
        self.addAlgorithm(EBMRegressionAlgorithm())
        self.addAlgorithm(EBMClassificationAlgorithm())
        self.addAlgorithm(ModelResidualAutocorrelationAlgorithm())
        self.addAlgorithm(PredictionUncertaintyMapAlgorithm())
        self.addAlgorithm(ConformalPredictionAlgorithm())
        self.addAlgorithm(TabPFNRegressionAlgorithm())
        self.addAlgorithm(TabPFNClassificationAlgorithm())
        self.addAlgorithm(DBSCANAlgorithm())
        self.addAlgorithm(HDBSCANAlgorithm())
        self.addAlgorithm(GMMClusteringAlgorithm())
