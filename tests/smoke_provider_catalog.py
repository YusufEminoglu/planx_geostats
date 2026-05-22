# -*- coding: utf-8 -*-
"""QGIS-independent smoke checks for the Processing provider catalog."""
from __future__ import annotations

import ast
import re
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "planx_geostats_provider.py"
MAIN_PLUGIN = ROOT / "main_plugin.py"
DEPENDENCIES = ROOT / "dependencies.py"
METADATA = ROOT / "metadata.txt"
CHANGELOG = ROOT / "CHANGELOG.md"
README = ROOT / "README.md"
QA_MATRIX = ROOT / "QA_MANUAL_TEST_MATRIX.md"
RELEASE_ZIP_VERIFIER = ROOT.parent / "packaging" / "verify_release_zip.py"
RELEASE_ZIP_VERIFIER_TEST = ROOT.parent / "packaging" / "test_verify_release_zip.py"
ALGORITHMS = ROOT / "algorithms"
ALGORITHM_ICONS = ROOT / "icons" / "algorithms"

EXPECTED_GROUPS = {
    "00 | Setup and Diagnostics",
    "01 | Data Preparation and Neighborhoods",
    "02 | Urban Pattern Scan",
    "03 | Hot Spots and Spatial Outliers",
    "04 | Centers, Direction and Dispersion",
    "05 | Models and Scenarios",
}

MIN_EXPECTED_ALGORITHM_COUNT = 32


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _provider_imported_algorithm_classes() -> set[str]:
    tree = _module_tree(PROVIDER)
    imported = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 1 or not node.module or not node.module.startswith("algorithms."):
            continue
        for alias in node.names:
            imported.add(alias.asname or alias.name)
    return imported


def _provider_registered_algorithm_classes() -> list[str]:
    tree = _module_tree(PROVIDER)
    registered = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "addAlgorithm":
            continue
        if not node.args:
            continue
        algorithm_call = node.args[0]
        if isinstance(algorithm_call, ast.Call) and isinstance(algorithm_call.func, ast.Name):
            registered.append(algorithm_call.func.id)
    return registered


def _algorithm_class_catalog() -> dict[str, dict[str, str]]:
    catalog = {}
    for path in sorted(ALGORITHMS.glob("alg_*.py")):
        tree = _module_tree(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith("Algorithm"):
                continue
            method_values = {}
            has_icon = False
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "icon":
                    has_icon = True
                if isinstance(child, ast.FunctionDef) and child.name in {"name", "displayName", "group"}:
                    method_values[child.name] = _single_string_return(child)
            catalog[node.name] = {
                "file": path.name,
                "name": method_values.get("name", ""),
                "display_name": method_values.get("displayName", ""),
                "group": method_values.get("group", ""),
                "has_icon": has_icon,
            }
    return catalog


def _single_string_return(function: ast.FunctionDef) -> str:
    for node in ast.walk(function):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return ""


def test_provider_imports_every_registered_algorithm() -> None:
    imported = _provider_imported_algorithm_classes()
    registered = _provider_registered_algorithm_classes()
    assert registered, "Provider should register at least one algorithm"
    assert len(registered) == len(set(registered)), "Provider should not register duplicate algorithm classes"
    missing_imports = sorted(set(registered) - imported)
    assert not missing_imports, f"Registered classes are not imported: {missing_imports}"


def test_every_algorithm_file_is_registered_once() -> None:
    catalog = _algorithm_class_catalog()
    registered = _provider_registered_algorithm_classes()
    assert len(registered) >= MIN_EXPECTED_ALGORITHM_COUNT, "Provider algorithm count unexpectedly decreased"
    unregistered = sorted(set(catalog) - set(registered))
    unknown = sorted(set(registered) - set(catalog))
    assert not unregistered, f"Algorithm classes missing from provider: {unregistered}"
    assert not unknown, f"Provider registers unknown algorithm classes: {unknown}"


def test_algorithm_catalog_has_stable_ids_and_groups() -> None:
    catalog = _algorithm_class_catalog()
    algorithm_ids = [meta["name"] for meta in catalog.values()]
    assert all(algorithm_ids), "Every algorithm needs a non-empty Processing id"
    assert len(algorithm_ids) == len(set(algorithm_ids)), "Processing ids must be unique"
    assert all(value == value.lower() for value in algorithm_ids), "Processing ids should stay lowercase"
    assert all(" " not in value and "-" not in value for value in algorithm_ids), "Processing ids should be import-safe"

    display_names = [meta["display_name"] for meta in catalog.values()]
    assert all(display_names), "Every algorithm needs a display name"

    groups = {meta["group"] for meta in catalog.values()}
    assert groups == EXPECTED_GROUPS


def test_every_algorithm_has_unique_png_icon() -> None:
    catalog = _algorithm_class_catalog()
    missing_methods = sorted(name for name, meta in catalog.items() if not meta["has_icon"])
    assert not missing_methods, f"Algorithm classes missing icon() methods: {missing_methods}"

    missing_icons = sorted(meta["name"] for meta in catalog.values() if not (ALGORITHM_ICONS / f"{meta['name']}.png").exists())
    assert not missing_icons, f"Algorithm PNG icons are missing: {missing_icons}"

    icon_files = sorted(path.stem for path in ALGORITHM_ICONS.glob("*.png"))
    unknown_icons = sorted(set(icon_files) - {meta["name"] for meta in catalog.values()})
    assert not unknown_icons, f"Unregistered algorithm icon files were found: {unknown_icons}"

    icon_hashes = {}
    for meta in catalog.values():
        path = ALGORITHM_ICONS / f"{meta['name']}.png"
        assert _png_dimensions(path) == (64, 64), f"Algorithm icon should be 64x64 PNG: {path.name}"
        checksum = zlib.crc32(path.read_bytes())
        assert checksum not in icon_hashes, f"Algorithm icons must be visually/file distinct: {path.name} matches {icon_hashes[checksum]}"
        icon_hashes[checksum] = path.name


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), f"Not a PNG file: {path.name}"
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def test_plugin_ui_surface_stays_processing_only() -> None:
    forbidden_terms = [
        "QAction",
        "QDialog",
        "QProcess",
        "QPushButton",
        "addPluginToMenu",
        "addToolBarIcon",
        "removePluginMenu",
        "removeToolBarIcon",
    ]
    combined = MAIN_PLUGIN.read_text(encoding="utf-8") + "\n" + DEPENDENCIES.read_text(encoding="utf-8")
    found = [term for term in forbidden_terms if term in combined]
    assert not found, f"Unexpected non-Processing UI hooks found: {found}"


def test_html_module_is_not_shadowed_in_report_writers() -> None:
    offenders = []
    for path in sorted(ALGORITHMS.glob("alg_*.py")):
        tree = _module_tree(path)
        imports_html = any(
            isinstance(node, ast.Import) and any(alias.name == "html" for alias in node.names)
            for node in tree.body
        )
        if not imports_html:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "html":
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"Do not shadow the imported html module: {offenders}"


def test_direct_polyline_polygon_calls_have_multipart_guard() -> None:
    offenders = []
    for path in sorted(ALGORITHMS.glob("alg_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        function_ranges = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                body = "\n".join(source.splitlines()[start - 1:end])
                function_ranges.append((start, end, body))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"asPolyline", "asPolygon"}:
                continue
            guarded = False
            for start, end, body in function_ranges:
                if start <= node.lineno <= end:
                    guarded = (
                        "isMultipart" in body
                        or "asMultiPolyline" in body
                        or "asMultiPolygon" in body
                    )
                    break
            if not guarded:
                offenders.append(f"{path.name}:{node.lineno}:{node.func.attr}")
    assert not offenders, f"Direct geometry conversion calls need multipart guards: {offenders}"


def test_release_documentation_version_is_synchronized() -> None:
    metadata_text = METADATA.read_text(encoding="utf-8")
    version_line = next(
        (line for line in metadata_text.splitlines() if line.startswith("version=")),
        "",
    )
    assert version_line, "metadata.txt must define a plugin version"
    version = version_line.split("=", 1)[1].strip()

    changelog_text = CHANGELOG.read_text(encoding="utf-8")
    readme_text = README.read_text(encoding="utf-8")
    assert f"## [{version}]" in changelog_text, "CHANGELOG.md top release should match metadata version"
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), "metadata version should use semantic version format"
    assert f"--version {version}" in readme_text, "README release verification command should match metadata version"
    assert f"{version} -" in metadata_text, "metadata changelog should include the current version entry"


def test_workflow_advisor_covers_core_decision_sections() -> None:
    advisor = ALGORITHMS / "alg_workflow_advisor.py"
    advisor_core = ROOT / "core" / "workflow_advisor.py"
    assert advisor.exists(), "Workflow Advisor algorithm should exist"
    assert advisor_core.exists(), "Workflow Advisor core recommendation helper should exist"
    source = advisor.read_text(encoding="utf-8") + "\n" + advisor_core.read_text(encoding="utf-8")
    required_terms = [
        "Planning Questions to Tool Sequences",
        "Personalized Recommendation",
        "Tool Selection Matrix",
        "Method Assumptions and Cautions",
        "Common Pitfalls and Safer Moves",
        "Starter Recipes for Bundled Samples",
        "Quality Gates",
        "Interpretation Discipline",
        "Data Readiness Audit",
        "Global Moran's I",
        "Getis-Ord Gi*",
        "Local Moran's I",
        "GWR; MGWR",
        "Spatial Lag Regression",
        "Spatial Error Regression",
        "Model Comparison Matrix",
        "median_land_surface_temp_c",
        "count_target",
        "qa_lines_directional",
        "QA_MANUAL_TEST_MATRIX.md",
        "GOAL_OPTIONS",
        "GEOMETRY_OPTIONS",
        "OUTCOME_OPTIONS",
        "personalized_recommendation",
        "Bundled sample fields or layers to try",
        "Combination warnings",
        "median_ndvi",
        "qa_ols_model_output",
    ]
    missing = [term for term in required_terms if term not in source]
    assert not missing, f"Workflow Advisor should cover core guidance terms: {missing}"


def test_model_comparison_uses_core_audit_helper() -> None:
    algorithm = ALGORITHMS / "alg_model_comparison.py"
    audit_core = ROOT / "core" / "model_audit.py"
    assert audit_core.exists(), "Model comparison scoring helper should exist"

    algorithm_source = algorithm.read_text(encoding="utf-8")
    core_source = audit_core.read_text(encoding="utf-8")
    assert "from ..core.model_audit import assign_model_scores, model_recommendation" in algorithm_source
    assert "def _assign_scores" not in algorithm_source, "Scoring should stay in the QGIS-independent core helper"
    required_terms = [
        "assign_model_scores",
        "residual_pattern_penalty",
        "model_recommendation",
        "without a strong global residual spatial pattern",
        "missing-diagnostic penalty",
    ]
    missing = [term for term in required_terms if term not in core_source + "\n" + algorithm_source]
    assert not missing, f"Model audit helper should cover scoring/report terms: {missing}"


def test_sensitivity_test_uses_core_audit_helper() -> None:
    algorithm = ALGORITHMS / "alg_sensitivity_test.py"
    audit_core = ROOT / "core" / "sensitivity_audit.py"
    assert audit_core.exists(), "Sensitivity test interpretation helper should exist"

    algorithm_source = algorithm.read_text(encoding="utf-8")
    core_source = audit_core.read_text(encoding="utf-8")
    assert "from ..core.sensitivity_audit import sensitivity_verdict" in algorithm_source
    required_terms = [
        "sensitivity_verdict",
        "sensitivity_next_action",
        "Sensitivity Cautions",
        "isolated observations",
        "Very dense neighborhood graph",
    ]
    missing = [term for term in required_terms if term not in core_source + "\n" + algorithm_source]
    assert not missing, f"Sensitivity audit helper should cover interpretation terms: {missing}"


def test_global_moran_uses_core_interpretation_helper() -> None:
    algorithm = ALGORITHMS / "alg_global_moran.py"
    audit_core = ROOT / "core" / "spatial_autocorrelation_audit.py"
    assert audit_core.exists(), "Global Moran interpretation helper should exist"

    algorithm_source = algorithm.read_text(encoding="utf-8")
    core_source = audit_core.read_text(encoding="utf-8")
    assert "from ..core.spatial_autocorrelation_audit import global_moran_interpretation" in algorithm_source
    required_terms = [
        "global_moran_interpretation",
        "evidence_strength",
        "global_moran_next_action",
        "Clustered",
        "Dispersed",
        "Local Moran's I or Gi*",
    ]
    missing = [term for term in required_terms if term not in core_source + "\n" + algorithm_source]
    assert not missing, f"Global Moran interpretation helper should cover report terms: {missing}"


def test_local_pattern_tools_use_core_summary_and_metadata_helpers() -> None:
    audit_core = ROOT / "core" / "local_pattern_audit.py"
    getis = ALGORITHMS / "alg_getis_ord.py"
    lisa = ALGORITHMS / "alg_local_moran.py"
    assert audit_core.exists(), "Local pattern class-summary helper should exist"

    combined = (
        audit_core.read_text(encoding="utf-8")
        + "\n"
        + getis.read_text(encoding="utf-8")
        + "\n"
        + lisa.read_text(encoding="utf-8")
    )
    required_terms = [
        "getis_ord_class_summary",
        "local_moran_class_summary",
        "Gi* classified",
        "Local Moran classified",
        "apply_output_metadata",
        "PlanX GeoStats Local Moran cluster and outlier output",
    ]
    missing = [term for term in required_terms if term not in combined]
    assert not missing, f"Local pattern tools should use class summaries and metadata helpers: {missing}"


def test_cluster_similarity_outputs_apply_metadata_aliases() -> None:
    clustering = ALGORITHMS / "alg_multivariate_clustering.py"
    similarity = ALGORITHMS / "alg_similarity_search.py"
    checks = {
        clustering: [
            "apply_output_metadata",
            "PlanX GeoStats multivariate clustering output",
            "cluster_id",
            "clust_size",
            "clust_dist",
        ],
        similarity: [
            "apply_output_metadata",
            "PlanX GeoStats similarity search output",
            "is_target",
            "sim_index",
            "sim_rank",
            "sim_pct",
            "sim_tier",
        ],
    }
    for path, required_terms in checks.items():
        source = path.read_text(encoding="utf-8")
        missing = [term for term in required_terms if term not in source]
        assert not missing, f"{path.name} should apply metadata aliases to analytical output fields: {missing}"


def test_model_output_layers_apply_metadata_aliases() -> None:
    checks = {
        ALGORITHMS / "alg_gwr.py": [
            "PlanX GeoStats GWR local model output",
            "y_predicted",
            "local_r2",
            "gwr_nbrs",
        ],
        ALGORITHMS / "alg_mgwr.py": [
            "PlanX GeoStats MGWR multiscale local model output",
            "mgwr_pred",
            "mgwr_std",
            "mgwr_used",
        ],
        ALGORITHMS / "alg_spatial_autoregression.py": [
            "PlanX GeoStats spatial lag regression output",
            "sar_pred",
            "sar_stdres",
            "sar_used",
        ],
        ALGORITHMS / "alg_spatial_error_regression.py": [
            "PlanX GeoStats spatial error regression output",
            "sem_pred",
            "sem_stdres",
            "sem_used",
        ],
    }
    for path, required_terms in checks.items():
        source = path.read_text(encoding="utf-8")
        assert "apply_output_metadata" in source, f"{path.name} should apply output metadata aliases"
        missing = [term for term in required_terms if term not in source]
        assert not missing, f"{path.name} should document model output fields with aliases: {missing}"


def test_center_direction_outputs_apply_metadata_aliases() -> None:
    checks = {
        ALGORITHMS / "alg_mean_center.py": [
            "PlanX GeoStats mean center output",
            "mean_x",
            "total_w",
            "skip_geom",
        ],
        ALGORITHMS / "alg_median_center.py": [
            "PlanX GeoStats median center output",
            "median_x",
            "total_dist",
        ],
        ALGORITHMS / "alg_standard_distance.py": [
            "PlanX GeoStats standard distance output",
            "std_dist",
            "radius",
            "input_n",
        ],
        ALGORITHMS / "alg_sde.py": [
            "PlanX GeoStats directional distribution output",
            "rotation",
            "semi_major",
            "semi_minor",
        ],
        ALGORITHMS / "alg_linear_directional_mean.py": [
            "PlanX GeoStats linear directional mean output",
            "mean_angle",
            "mean_length",
            "line_count",
        ],
        ALGORITHMS / "alg_central_feature.py": [
            "PlanX GeoStats central feature output",
            "is_central",
            "total_distance",
        ],
    }
    for path, required_terms in checks.items():
        source = path.read_text(encoding="utf-8")
        assert "postProcessAlgorithm" in source, f"{path.name} should apply metadata in post-processing"
        assert "apply_output_metadata" in source, f"{path.name} should apply output metadata aliases"
        missing = [term for term in required_terms if term not in source]
        assert not missing, f"{path.name} should document center/direction output fields with aliases: {missing}"


def test_manual_qa_matrix_covers_release_workflows() -> None:
    assert QA_MATRIX.exists(), "Manual QA matrix should exist"
    content = QA_MATRIX.read_text(encoding="utf-8")
    required_terms = [
        "GeoStats Workflow Advisor",
        "Data Readiness Audit",
        "Global Moran's I",
        "Getis-Ord Gi*",
        "Linear Directional Mean",
        "Center/direction output metadata",
        "OLS Regression",
        "Model Comparison Matrix",
        "Report Decision Engines",
        "Workflow Advisor recommendation engine",
        "Model Comparison audit engine",
        "Sensitivity interpretation engine",
        "Global Moran interpretation engine",
        "Local pattern class-summary engine",
        "Release Gate",
    ]
    missing = [term for term in required_terms if term not in content]
    assert not missing, f"Manual QA matrix should cover core workflows: {missing}"


def test_readme_documents_core_decision_helpers_and_release_zip_gate() -> None:
    content = README.read_text(encoding="utf-8")
    required_terms = [
        "QGIS-independent core helpers",
        "workflow advising",
        "model-comparison scoring",
        "Monte Carlo sensitivity interpretation",
        "Global Moran's I report interpretation",
        "developer-only paths are absent",
        "algorithm icons are present",
        "Processing-only",
    ]
    missing = [term for term in required_terms if term not in content]
    assert not missing, f"README should document decision-helper and release-zip gates: {missing}"


def test_optional_dependency_failures_use_shared_guidance() -> None:
    dependency_helper = ROOT / "dependencies.py"
    assert "def optional_dependency_error" in dependency_helper.read_text(encoding="utf-8")
    checks = {
        ALGORITHMS / "alg_mgwr.py": ["optional_dependency_error", "Multiscale Geographically Weighted Regression", "mgwr"],
        ALGORITHMS / "alg_spatial_autoregression.py": ["optional_dependency_error", "Spatial Autoregression", "libpysal", "spreg"],
        ALGORITHMS / "alg_spatial_error_regression.py": ["optional_dependency_error", "Spatial Error Regression", "libpysal", "spreg"],
    }
    for path, required_terms in checks.items():
        source = path.read_text(encoding="utf-8")
        missing = [term for term in required_terms if term not in source]
        assert not missing, f"{path.name} should use shared optional dependency guidance: {missing}"


def test_release_zip_verifier_guards_geostats_packaging_contract() -> None:
    assert RELEASE_ZIP_VERIFIER.exists(), "Release zip verifier should exist"
    assert RELEASE_ZIP_VERIFIER_TEST.exists(), "Release zip verifier smoke test should exist"
    content = RELEASE_ZIP_VERIFIER.read_text(encoding="utf-8")
    test_content = RELEASE_ZIP_VERIFIER_TEST.read_text(encoding="utf-8")
    required_terms = [
        "MIN_ALGORITHM_ICON_COUNT",
        "icons/algorithms",
        "metadata icon is missing from zip",
        "Forbidden development path in zip",
        "Unexpected non-Processing UI hook",
        "addPluginToMenu",
        "addToolBarIcon",
    ]
    missing = [term for term in required_terms if term not in content]
    assert not missing, f"Release zip verifier should guard packaging contract: {missing}"
    test_terms = [
        "test_valid_processing_zip_passes",
        "test_processing_zip_rejects_ui_hooks_and_missing_algorithm_icons",
        "test_non_processing_zip_allows_toolbar_style_plugin",
        "test_forbidden_source_artifacts_are_rejected",
    ]
    missing_tests = [term for term in test_terms if term not in test_content]
    assert not missing_tests, f"Release zip verifier smoke tests should cover key scenarios: {missing_tests}"


def test_professional_report_helpers_are_used_by_key_reports() -> None:
    reporting = ROOT / "core" / "reporting.py"
    metadata_helper = ROOT / "core" / "layer_metadata.py"
    assert reporting.exists(), "Shared reporting helper should exist"
    assert metadata_helper.exists(), "Layer metadata helper should exist"

    key_reports = [
        ALGORITHMS / "alg_spatial_regression.py",
        ALGORITHMS / "alg_global_moran.py",
        ALGORITHMS / "alg_generalized_linear_regression.py",
    ]
    for path in key_reports:
        source = path.read_text(encoding="utf-8")
        assert "analyst_guidance_html" in source, f"{path.name} should use shared analyst guidance"
        assert "analyst_guidance_css" in source, f"{path.name} should use shared analyst guidance CSS"

    output_layers = [
        ALGORITHMS / "alg_spatial_regression.py",
        ALGORITHMS / "alg_getis_ord.py",
    ]
    for path in output_layers:
        source = path.read_text(encoding="utf-8")
        assert "apply_output_metadata" in source, f"{path.name} should apply output metadata aliases"


def run_all() -> None:
    test_provider_imports_every_registered_algorithm()
    test_every_algorithm_file_is_registered_once()
    test_algorithm_catalog_has_stable_ids_and_groups()
    test_every_algorithm_has_unique_png_icon()
    test_plugin_ui_surface_stays_processing_only()
    test_html_module_is_not_shadowed_in_report_writers()
    test_direct_polyline_polygon_calls_have_multipart_guard()
    test_release_documentation_version_is_synchronized()
    test_workflow_advisor_covers_core_decision_sections()
    test_model_comparison_uses_core_audit_helper()
    test_sensitivity_test_uses_core_audit_helper()
    test_global_moran_uses_core_interpretation_helper()
    test_local_pattern_tools_use_core_summary_and_metadata_helpers()
    test_cluster_similarity_outputs_apply_metadata_aliases()
    test_model_output_layers_apply_metadata_aliases()
    test_center_direction_outputs_apply_metadata_aliases()
    test_manual_qa_matrix_covers_release_workflows()
    test_readme_documents_core_decision_helpers_and_release_zip_gate()
    test_optional_dependency_failures_use_shared_guidance()
    test_release_zip_verifier_guards_geostats_packaging_contract()
    test_professional_report_helpers_are_used_by_key_reports()
    print("PROVIDER CATALOG SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
