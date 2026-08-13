# -*- coding: utf-8 -*-
"""Shared helpUrl mixin for all GeoStats algorithms."""

DOC_BASE_URL = "https://yusufeminoglu.github.io/planx_geostats/GEOSTATS_REFERENCE_MANUAL.html"

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
}


class HelpUrlMixin:
    def helpUrl(self) -> str:
        return DOC_BASE_URL + "#" + HELP_SLUGS[self.name()]
