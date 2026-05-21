# -*- coding: utf-8 -*-
"""Smoke checks for bundled PlanX GeoStats sample data."""
from __future__ import annotations

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data" / "planx_geostats_izmir_neighborhoods.gpkg"
LAYER = "planx_geostats_izmir_neighborhoods"


def run_all() -> None:
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

        count = con.execute(f'select count(*) as n from "{LAYER}"').fetchone()["n"]
        assert int(count) == 237

        columns = {row["name"] for row in con.execute(f'pragma table_info("{LAYER}")')}
        column_info = {row["name"]: row for row in con.execute(f'pragma table_info("{LAYER}")')}
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
            missing = con.execute(f'select count(*) as n from "{LAYER}" where "{field}" is null').fetchone()["n"]
            assert int(missing) == 0, f"{field} should be complete in the bundled sample"

        range_checks = {
            "median_ndvi": (-1.0, 1.0),
            "building_coverage_pct": (0.0, 150.0),
            "building_volume_density_pct": (0.0, 100.0),
            "park_m2_per_capita": (0.0, 100000.0),
            "median_land_surface_temp_c": (-30.0, 80.0),
        }
        for field, (lower, upper) in range_checks.items():
            row = con.execute(
                f'select min("{field}") as mn, max("{field}") as mx from "{LAYER}" where "{field}" is not null'
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
            unique_count = con.execute(
                f'select count(distinct "{field}") as n from "{LAYER}" where "{field}" is not null'
            ).fetchone()["n"]
            assert int(unique_count) >= 10, f"{field} lacks enough variation for sample workflows"

        street_unique = con.execute(
            f'select count(distinct street_connectivity) as n from "{LAYER}" where street_connectivity is not null'
        ).fetchone()["n"]
        assert int(street_unique) >= 4, "street_connectivity should retain ordinal network variation"

        density_classes = {
            row["urban_density_class"]
            for row in con.execute(f'select distinct urban_density_class from "{LAYER}" where urban_density_class is not null')
        }
        assert density_classes.issubset({"Low", "Moderate", "High", "Very High"})
        assert len(density_classes) >= 3
    finally:
        con.close()
    print("SAMPLE DATA SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
