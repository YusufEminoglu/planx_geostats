# -*- coding: utf-8 -*-
"""Shared helpUrl mixin for all GeoStats algorithms."""

# Canonical public home of the reference manual. Served from the plugin's
# docs/ directory by packaging/build_web_portal.py; metadata.txt's "about"
# field links to the same URL and QGIS Hub validates it on upload. Do not
# point this at a host that is not the one metadata.txt advertises - every
# algorithm's Help button resolve through here.
DOC_BASE_URL = "https://geophilo.com/planx_geostats/manual.html"

# GEOSTATS_REFERENCE_MANUAL.html uses hand-picked short anchor slugs that are
# not derivable from name() (e.g. "getis_ord_gi" -> "alg-getis-ord", not
# "alg-getis-ord-gi"). Keep this in sync with the manual's <article
# class="alg-card" id="..."> ids; anchor-id cross-check verifies 1:1 coverage.
HELP_SLUGS = {
    "geostats_library_status": "alg-library-status",
    "install_geostats_libraries": "alg-install-libraries",
    "sample_dataset_guide": "alg-sample-data",
    "data_readiness_audit": "alg-data-readiness",
    "geostats_workflow_advisor": "alg-workflow-advisor",
    "calculate_distance_band": "alg-distance-band",
    "export_attributes_to_ascii": "alg-export-attrs",
    "global_moran_autocorrelation": "alg-global-moran",
    "spatial_gini_inequality": "alg-spatial-gini",
    "general_g_autocorrelation": "alg-general-g",
    "incremental_spatial_autocorrelation": "alg-inc-autocorr",
    "ripleys_k_function": "alg-ripleys-k",
    "average_nearest_neighbor": "alg-ann",
    "getis_ord_gi": "alg-getis-ord",
    "local_moran_lisa": "alg-local-moran",
    "bivariate_lisa": "alg-bivariate-lisa",
    "bivariate_spatial_association_lees_l": "alg-bivariate-lee",
    "multivariate_clustering": "alg-multivar-cluster",
    "similarity_search": "alg-similarity",
    "mean_center": "alg-mean-center",
    "central_feature": "alg-central-feature",
    "median_center": "alg-median-center",
    "standard_distance": "alg-std-distance",
    "directional_distribution": "alg-sde",
    "linear_directional_mean": "alg-linear-dir-mean",
    "ols_regression": "alg-ols",
    "generalized_linear_regression": "alg-glr",
    "spatial_autoregression": "alg-spatial-lag",
    "spatial_error_regression": "alg-spatial-error",
    "exploratory_regression": "alg-exploratory-reg",
    "gwr_regression": "alg-gwr",
    "multiscale_geographically_weighted_regression": "alg-mgwr",
    "sensitivity_test": "alg-sensitivity",
    "model_comparison_matrix": "alg-model-comparison",
    "geary_c_autocorrelation": "alg-geary-c",
    "join_count_statistics": "alg-join-count",
    "global_bivariate_lee_l": "alg-global-lee-l",
    "geodetector_q_statistic": "alg-geodetector-q",
    "local_geary_c": "alg-local-geary-c",
    "colocation_quotient": "alg-colocation-quotient",
    "skater_regionalization": "alg-skater",
    "lm_diagnostics": "alg-lm-diagnostics",
    "spatial_durbin_model": "alg-spatial-durbin",
    "esf_regression": "alg-esf",
    "random_forest_regression": "alg-rf-regression",
    "random_forest_classification": "alg-rf-classification",
    "extra_trees_regression": "alg-et-regression",
    "extra_trees_classification": "alg-et-classification",
    "support_vector_regression": "alg-svr",
    "support_vector_classification": "alg-svc",
    "gbm_regression_sklearn": "alg-gbm-reg-sklearn",
    "gbm_regression_xgboost": "alg-gbm-reg-xgboost",
    "gbm_regression_lightgbm": "alg-gbm-reg-lightgbm",
    "gbm_classification_sklearn": "alg-gbm-cls-sklearn",
    "gbm_classification_xgboost": "alg-gbm-cls-xgboost",
    "gbm_classification_lightgbm": "alg-gbm-cls-lightgbm",
    "gbm_regression_catboost": "alg-gbm-reg-catboost",
    "gbm_classification_catboost": "alg-gbm-cls-catboost",
    "mlp_regression": "alg-mlp-regression",
    "mlp_classification": "alg-mlp-classification",
    "spatial_kfold_cv_evaluator": "alg-spatial-cv",
    "permutation_feature_importance": "alg-permutation-importance",
    "partial_dependence_report": "alg-partial-dependence",
    "ml_model_comparison": "alg-ml-model-comparison",
    "shap_global_importance": "alg-shap-global",
    "shap_spatial_map": "alg-shap-spatial-map",
    "shap_local_explanation": "alg-shap-local",
    "model_residual_autocorrelation": "alg-model-residual-autocorr",
    "prediction_uncertainty_map": "alg-prediction-uncertainty",
    "dbscan_clustering": "alg-dbscan",
    "hdbscan_clustering": "alg-hdbscan",
    "gmm_clustering": "alg-gmm",
    "spatial_regime_regression": "alg-spatial-regime",
    "quantile_regression": "alg-quantile-regression",
    "gw_summary_statistics": "alg-gw-summary-stats",
    "conformal_prediction_interval": "alg-conformal-prediction",
    "tabpfn_regression": "alg-tabpfn-regression",
    "tabpfn_classification": "alg-tabpfn-classification",
    "dice_counterfactual_explanation": "alg-dice-counterfactual",
    "ebm_regression": "alg-ebm-regression",
    "ebm_classification": "alg-ebm-classification",
    "empirical_semivariogram": "alg-semivariogram",
    "kriging_interpolation": "alg-kriging",
    "idw_interpolation": "alg-idw",
    "rbf_interpolation": "alg-rbf",
    "natural_neighbour_interpolation": "alg-natural-neighbour",
    "ebk_interpolation": "alg-ebk",
    "polynomial_trend_surface": "alg-trend-surface",
    "indicator_kriging": "alg-indicator",
    "interp_compare": "alg-interp-compare",
    "rate_smoothing": "alg-rate-smoothing",
    "eb_rate_lisa": "alg-eb-rate-lisa",
    "multivariate_local_geary": "alg-mv-local-geary",
    "local_join_counts": "alg-local-join-counts",
    "losh_heteroskedasticity": "alg-losh",
    "local_partial_moran": "alg-partial-moran",
    "gamma_index": "alg-gamma",
    "smaup_test": "alg-smaup",
    "max_p_regions": "alg-max-p",
    "azp_regionalization": "alg-azp",
    "redcap_regionalization": "alg-redcap",
    "regionalization_quality": "alg-region-quality",
}


# Search words for the Processing toolbox (and any search that reads tags()).
# 3.12 defined no tags, so "hotspot" found nothing although it is the word most
# users type.  Every tool gets the family words; the ones below add their own.
COMMON_TAGS = ["geostats", "spatial statistics", "planx"]
_HOTSPOT = ["hotspot", "hot spot", "hot spots", "hotspots", "cold spot", "cluster", "clustering"]
_LOCAL = ["local spatial autocorrelation", "spatial autocorrelation", "spatial weights",
          "pysal", "esda", "geoda", "local indicators of spatial association"]
_GLOBAL = ["spatial autocorrelation", "global spatial autocorrelation", "spatial weights",
           "pysal", "esda", "geoda", "clustering", "dispersion"]
_REGION = ["regionalization", "regionalisation", "spatially constrained clustering", "zoning",
           "spatial weights", "pysal", "spopt", "cluster"]
_REGRESSION = ["regression", "spatial regression", "spatial econometrics", "pysal", "spreg"]
ALGORITHM_TAGS = {
    "getis_ord_gi": _HOTSPOT + _LOCAL + ["gi*", "gi star", "getis-ord", "getis ord", "hot spot analysis",
                                         "fdr", "false discovery rate"],
    "local_moran_lisa": _HOTSPOT + _LOCAL + ["lisa", "local moran", "local moran's i", "moran", "anselin",
                                             "outlier", "spatial outlier", "cluster and outlier", "fdr"],
    "bivariate_lisa": _HOTSPOT + _LOCAL + ["lisa", "bivariate lisa", "bivariate moran", "local moran", "outlier"],
    "local_geary_c": _HOTSPOT + _LOCAL + ["lisa", "local geary", "geary", "outlier"],
    "eb_rate_lisa": _HOTSPOT + _LOCAL + ["lisa", "local moran", "empirical bayes", "rate", "disease mapping"],
    "local_partial_moran": _HOTSPOT + _LOCAL + ["lisa", "local moran", "partial moran", "conditional"],
    "multivariate_local_geary": _HOTSPOT + _LOCAL + ["lisa", "local geary", "multivariate"],
    "local_join_counts": _HOTSPOT + _LOCAL + ["lisa", "join count", "binary", "categorical"],
    "losh_heteroskedasticity": _LOCAL + ["losh", "local spatial heteroskedasticity", "hot spot", "volatility"],
    "bivariate_spatial_association_lees_l": _LOCAL + ["lee's l", "lees l", "bivariate", "lisa", "cluster"],
    "global_moran_autocorrelation": _GLOBAL + ["moran", "moran's i", "global moran"],
    "geary_c_autocorrelation": _GLOBAL + ["geary", "geary's c"],
    "general_g_autocorrelation": _GLOBAL + ["general g", "getis-ord", "high/low clustering", "hot spot"],
    "join_count_statistics": _GLOBAL + ["join count", "binary", "categorical"],
    "global_bivariate_lee_l": _GLOBAL + ["lee's l", "bivariate"],
    "incremental_spatial_autocorrelation": _GLOBAL + ["moran", "distance band", "scale", "incremental"],
    "spatial_gini_inequality": ["gini", "inequality", "spatial gini", "spatial autocorrelation", "pysal"],
    "gamma_index": _GLOBAL + ["gamma"],
    "calculate_distance_band": ["spatial weights", "distance band", "knn", "queen", "rook", "contiguity",
                                "neighbours", "neighbors", "pysal", "libpysal"],
    "skater_regionalization": _REGION + ["skater"],
    "max_p_regions": _REGION + ["max-p", "max p", "maxp"],
    "azp_regionalization": _REGION + ["azp", "automatic zoning"],
    "redcap_regionalization": _REGION + ["redcap"],
    "regionalization_quality": _REGION + ["silhouette", "compactness"],
    "ols_regression": _REGRESSION + ["ols", "ordinary least squares"],
    "spatial_autoregression": _REGRESSION + ["spatial lag", "sar"],
    "spatial_error_regression": _REGRESSION + ["spatial error", "sem"],
    "spatial_durbin_model": _REGRESSION + ["spatial durbin", "sdm"],
    "lm_diagnostics": _REGRESSION + ["lagrange multiplier", "lm test", "spatial dependence"],
    "esf_regression": _REGRESSION + ["eigenvector spatial filtering", "esf", "moran eigenvectors"],
    "gwr_regression": _REGRESSION + ["gwr", "geographically weighted regression", "mgwr"],
    "multiscale_geographically_weighted_regression": _REGRESSION + ["mgwr", "gwr", "multiscale"],
}


class HelpUrlMixin:
    def helpUrl(self) -> str:
        return DOC_BASE_URL + "#" + HELP_SLUGS[self.name()]

    def tags(self):
        """Search words: the family words plus this tool's own (see ALGORITHM_TAGS)."""
        words = list(ALGORITHM_TAGS.get(self.name(), [])) + COMMON_TAGS
        seen = set()
        return [w for w in words if not (w in seen or seen.add(w))]
