# -*- coding: utf-8 -*-
"""Smoke checks for bundled PlanX GeoStats sample data."""
from __future__ import annotations

import sqlite3
import struct
import ast
from pathlib import Path

import numpy as np

import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "planx_geostats_izmir_neighborhoods.gpkg"
SYNTHETIC_QA = ROOT / "sample_data" / "planx_geostats_synthetic_qa.gpkg"
SAMPLE_GUIDE = ROOT / "algorithms" / "alg_sample_data_guide.py"
LAYER = "planx_geostats_izmir_neighborhoods"


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


stats = load_module("stats_engines_for_sample", "core/stats_engines.py")


def quote_identifier(value: str, allowed: set[str]) -> str:
    assert value in allowed, f"Unexpected SQL identifier: {value}"
    return '"' + value.replace('"', '""') + '"'


def run_all() -> None:
    test_izmir_sample_data()
    test_synthetic_qa_sample_data()
    test_sample_guide_load_options_match_bundled_layers()
    test_sample_guide_html_mentions_all_loadable_layers()
    print("SAMPLE DATA SMOKE TESTS OK")


def test_izmir_sample_data() -> None:
    assert SAMPLE.exists(), f"Missing sample GeoPackage: {SAMPLE}"
    con = sqlite3.connect(SAMPLE)
    con.row_factory = sqlite3.Row
    try:
        content = con.execute(
            "select table_name, data_type, srs_id from gpkg_contents where table_name = ?",
            (LAYER,),
        ).fetchone()
        assert content is not None
        assert content["data_type"] == "features"
        assert int(content["srs_id"]) == 5253

        geometry = con.execute(
            "select column_name, geometry_type_name from gpkg_geometry_columns where table_name = ?",
            (LAYER,),
        ).fetchone()
        assert geometry is not None
        assert geometry["column_name"] == "geom"
        assert geometry["geometry_type_name"].upper() == "POLYGON"

        allowed_layers = {LAYER}
        layer_sql = quote_identifier(LAYER, allowed_layers)
        count = con.execute(f"select count(*) as n from {layer_sql}").fetchone()["n"]
        assert int(count) == 237

        columns = {row["name"] for row in con.execute(f"pragma table_info({layer_sql})")}
        column_info = {row["name"]: row for row in con.execute(f"pragma table_info({layer_sql})")}
        required = {
            "neighborhood_name",
            "district_name",
            "official_population",
            "median_heat_island_index",
            "median_land_surface_temp_c",
            "median_ndvi",
            "building_coverage_pct",
            "park_m2_per_capita",
            "street_connectivity",
            "normalized_integration",
            "urban_density_class",
        }
        assert required.issubset(columns)
        assert all(" " not in name and "-" not in name for name in columns)

        numeric_required = {
            "official_population",
            "median_heat_island_index",
            "median_land_surface_temp_c",
            "median_ndvi",
            "building_coverage_pct",
            "park_m2_per_capita",
            "street_connectivity",
            "normalized_integration",
            "neighborhood_area_m2",
        }
        for field in numeric_required:
            assert field in column_info, f"Missing numeric field: {field}"
            assert column_info[field]["type"].upper() in {"INTEGER", "REAL", "FLOAT", "DOUBLE"}, field

        text_required = {"neighborhood_name", "district_name", "urban_density_class"}
        for field in text_required:
            assert column_info[field]["type"].upper() in {"TEXT", "VARCHAR"}, field

        critical_complete = [
            "median_heat_island_index",
            "median_land_surface_temp_c",
            "median_ndvi",
            "building_coverage_pct",
            "street_connectivity",
            "normalized_integration",
        ]
        for field in critical_complete:
            field_sql = quote_identifier(field, columns)
            missing = con.execute(f"select count(*) as n from {layer_sql} where {field_sql} is null").fetchone()["n"]
            assert int(missing) == 0, f"{field} should be complete in the bundled sample"

        range_checks = {
            "median_ndvi": (-1.0, 1.0),
            "building_coverage_pct": (0.0, 150.0),
            "building_volume_density_pct": (0.0, 100.0),
            "park_m2_per_capita": (0.0, 100000.0),
            "median_land_surface_temp_c": (-30.0, 80.0),
        }
        for field, (lower, upper) in range_checks.items():
            field_sql = quote_identifier(field, columns)
            row = con.execute(
                f"select min({field_sql}) as mn, max({field_sql}) as mx from {layer_sql} where {field_sql} is not null"
            ).fetchone()
            assert row["mn"] is not None, f"{field} has no valid values"
            assert lower <= float(row["mn"]) <= upper, f"{field} minimum is outside expected range"
            assert lower <= float(row["mx"]) <= upper, f"{field} maximum is outside expected range"

        variation_fields = [
            "median_heat_island_index",
            "median_land_surface_temp_c",
            "median_ndvi",
            "park_m2_per_capita",
            "normalized_integration",
            "normalized_choice",
        ]
        for field in variation_fields:
            field_sql = quote_identifier(field, columns)
            unique_count = con.execute(
                f"select count(distinct {field_sql}) as n from {layer_sql} where {field_sql} is not null"
            ).fetchone()["n"]
            assert int(unique_count) >= 10, f"{field} lacks enough variation for sample workflows"

        street_sql = quote_identifier("street_connectivity", columns)
        street_unique = con.execute(
            f"select count(distinct {street_sql}) as n from {layer_sql} where {street_sql} is not null"
        ).fetchone()["n"]
        assert int(street_unique) >= 4, "street_connectivity should retain ordinal network variation"

        density_sql = quote_identifier("urban_density_class", columns)
        density_classes = {
            row["urban_density_class"]
            for row in con.execute(
                f"select distinct {density_sql} as urban_density_class from {layer_sql} where {density_sql} is not null"
            )
        }
        assert density_classes.issubset({"Low", "Moderate", "High", "Very High"})
        assert len(density_classes) >= 3
    finally:
        con.close()


def test_synthetic_qa_sample_data() -> None:
    assert SYNTHETIC_QA.exists(), f"Missing synthetic QA GeoPackage: {SYNTHETIC_QA}"
    con = sqlite3.connect(SYNTHETIC_QA)
    con.row_factory = sqlite3.Row
    try:
        expected_layers = {
            "qa_points_grid": ("POINT", 25, {
                "target_value", "explanatory_a", "explanatory_b", "binary_target", "count_target", "cluster_hint",
            }),
            "qa_lines_directional": ("GEOMETRY", 6, {"line_weight", "expected_bearing"}),
            "qa_polygons_mini": ("POLYGON", 9, {"target_value", "count_target"}),
            "qa_ols_model_output": ("POINT", 8, {"observed_y", "residual", "std_res"}),
            "qa_glr_model_output": ("POINT", 8, {"observed_y", "glr_fit", "glr_resid", "glr_used"}),
            "qa_gwr_model_output": ("POINT", 8, {"observed_y", "y_predicted", "residual"}),
            "qa_sar_model_output": ("POINT", 8, {"observed_y", "sar_pred", "sar_resid", "sar_used", "sar_stdres"}),
            "qa_sem_model_output": ("POINT", 8, {"observed_y", "sem_pred", "sem_resid", "sem_used", "sem_stdres"}),
            "qa_mgwr_model_output": ("POINT", 8, {"observed_y", "mgwr_pred", "mgwr_resid", "mgwr_used", "mgwr_std"}),
        }
        contents = {
            row["table_name"]: row
            for row in con.execute("select table_name, data_type, srs_id from gpkg_contents")
        }
        assert set(expected_layers).issubset(contents)
        allowed_layers = set(expected_layers)
        for layer, (geometry_type, expected_count, required_fields) in expected_layers.items():
            assert contents[layer]["data_type"] == "features"
            assert int(contents[layer]["srs_id"]) == 3857

            geometry = con.execute(
                "select column_name, geometry_type_name from gpkg_geometry_columns where table_name = ?",
                (layer,),
            ).fetchone()
            assert geometry is not None
            assert geometry["column_name"] == "geom"
            assert geometry["geometry_type_name"].upper() == geometry_type

            layer_sql = quote_identifier(layer, allowed_layers)
            count = con.execute(f"select count(*) as n from {layer_sql}").fetchone()["n"]
            assert int(count) == expected_count, f"{layer} feature count changed"

            columns = {row["name"] for row in con.execute(f"pragma table_info({layer_sql})")}
            assert required_fields.issubset(columns), f"{layer} is missing required QA fields"

        points_sql = quote_identifier("qa_points_grid", allowed_layers)
        binary_values = {
            row["binary_target"]
            for row in con.execute(f"select distinct binary_target from {points_sql}")
        }
        assert binary_values == {0, 1}
        count_values = [
            row["count_target"]
            for row in con.execute(f"select count_target from {points_sql}")
        ]
        assert min(count_values) >= 0
        assert len(set(count_values)) >= 6

        model_layers = [name for name in expected_layers if name.endswith("_model_output")]
        for layer in model_layers:
            layer_sql = quote_identifier(layer, allowed_layers)
            missing_obs = con.execute(f"select count(*) as n from {layer_sql} where observed_y is null").fetchone()["n"]
            assert int(missing_obs) == 0, f"{layer} should have complete observed_y values"

        point_rows = con.execute(
            f"select geom, target_value, explanatory_a, explanatory_b, binary_target, count_target from {points_sql}"
        ).fetchall()
        xs, ys = [], []
        target, exp_a, exp_b, binary, counts = [], [], [], [], []
        for row in point_rows:
            x, y = _read_gpkg_point(row["geom"])
            xs.append(x)
            ys.append(y)
            target.append(float(row["target_value"]))
            exp_a.append(float(row["explanatory_a"]))
            exp_b.append(float(row["explanatory_b"]))
            binary.append(float(row["binary_target"]))
            counts.append(float(row["count_target"]))
        x_arr = np.array(xs, dtype=float)
        y_arr = np.array(ys, dtype=float)
        ann = stats.calculate_average_nearest_neighbor(x_arr, y_arr, study_area=1_000_000.0)
        assert len(ann) == 6
        assert all(np.isfinite(float(value)) for value in ann)

        ripley = stats.calculate_ripleys_k(x_arr, y_arr, 250.0, 250.0, 3, study_area=1_000_000.0)
        assert len(ripley) == 3
        assert all("l_minus_d" in row for row in ripley)

        x_data = np.column_stack((np.array(exp_a, dtype=float), np.array(exp_b, dtype=float)))
        logistic = stats.calculate_glr(np.array(binary, dtype=float), x_data, "logistic")
        poisson = stats.calculate_glr(np.array(counts, dtype=float), x_data, "poisson")
        assert len(logistic["fitted"]) == len(point_rows)
        assert len(poisson["fitted"]) == len(point_rows)

        lines_sql = quote_identifier("qa_lines_directional", allowed_layers)
        line_rows = con.execute(f"select geom from {lines_sql}").fetchall()
        starts_x, starts_y, ends_x, ends_y = [], [], [], []
        for row in line_rows:
            start, end = _read_first_line_endpoints(row["geom"])
            starts_x.append(start[0])
            starts_y.append(start[1])
            ends_x.append(end[0])
            ends_y.append(end[1])
        center_x, center_y, mean_angle, mean_length = stats.calculate_linear_directional_mean(
            np.array(starts_x),
            np.array(starts_y),
            np.array(ends_x),
            np.array(ends_y),
        )
        assert np.isfinite(center_x)
        assert np.isfinite(center_y)
        assert 0.0 <= mean_angle <= 360.0
        assert mean_length > 0.0
    finally:
        con.close()


def test_sample_guide_load_options_match_bundled_layers() -> None:
    constants = _sample_guide_constants()
    assert constants["LOAD_OPTIONS"] == [
        "Izmir planning sample",
        "Synthetic QA fixture",
        "Both datasets",
    ]
    assert constants["LAYER_NAME"] == LAYER

    con = sqlite3.connect(SYNTHETIC_QA)
    try:
        qa_layers = {
            row[0]
            for row in con.execute("select table_name from gpkg_contents where data_type = 'features'")
        }
    finally:
        con.close()

    listed_layers = set(constants["SYNTHETIC_QA_LAYERS"])
    assert listed_layers == qa_layers, "Sample Dataset Guide synthetic layer list must match the QA fixture"


def test_sample_guide_html_mentions_all_loadable_layers() -> None:
    source = SAMPLE_GUIDE.read_text(encoding="utf-8")
    constants = _sample_guide_constants()
    required_phrases = [
        "Loading Modes",
        "Izmir planning sample",
        "Synthetic QA fixture",
        "Both datasets",
        LAYER,
    ]
    required_phrases.extend(constants["SYNTHETIC_QA_LAYERS"])

    missing = [phrase for phrase in required_phrases if phrase not in source]
    assert not missing, f"Sample Dataset Guide HTML should mention every loadable layer/mode: {missing}"


def _sample_guide_constants() -> dict[str, object]:
    tree = ast.parse(SAMPLE_GUIDE.read_text(encoding="utf-8"), filename=str(SAMPLE_GUIDE))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SampleDataGuideAlgorithm":
            constants = {}
            for child in node.body:
                if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                    if child.targets[0].id in {"LOAD_OPTIONS", "LAYER_NAME", "SYNTHETIC_QA_LAYERS"}:
                        constants[child.targets[0].id] = ast.literal_eval(child.value)
            return constants
    raise AssertionError("SampleDataGuideAlgorithm class was not found")


def _read_gpkg_wkb(blob: bytes) -> bytes:
    assert blob[:2] == b"GP"
    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    return blob[8 + envelope_sizes[envelope_code]:]


def _read_gpkg_point(blob: bytes) -> tuple[float, float]:
    wkb = _read_gpkg_wkb(blob)
    byte_order = "<" if wkb[0] == 1 else ">"
    geom_type = struct.unpack(byte_order + "I", wkb[1:5])[0]
    assert geom_type == 1
    return struct.unpack(byte_order + "dd", wkb[5:21])


def _read_first_line_endpoints(blob: bytes) -> tuple[tuple[float, float], tuple[float, float]]:
    wkb = _read_gpkg_wkb(blob)
    byte_order = "<" if wkb[0] == 1 else ">"
    geom_type = struct.unpack(byte_order + "I", wkb[1:5])[0]
    if geom_type == 2:
        offset = 5
        n_points = struct.unpack(byte_order + "I", wkb[offset:offset + 4])[0]
        offset += 4
    else:
        assert geom_type == 5
        n_lines = struct.unpack(byte_order + "I", wkb[5:9])[0]
        assert n_lines > 0
        offset = 9
        assert wkb[offset] in {0, 1}
        line_order = "<" if wkb[offset] == 1 else ">"
        assert struct.unpack(line_order + "I", wkb[offset + 1:offset + 5])[0] == 2
        byte_order = line_order
        offset += 5
        n_points = struct.unpack(byte_order + "I", wkb[offset:offset + 4])[0]
        offset += 4
    assert n_points >= 2
    start = struct.unpack(byte_order + "dd", wkb[offset:offset + 16])
    end_offset = offset + (n_points - 1) * 16
    end = struct.unpack(byte_order + "dd", wkb[end_offset:end_offset + 16])
    return start, end


if __name__ == "__main__":
    run_all()
