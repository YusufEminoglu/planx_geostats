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
    finally:
        con.close()
    print("SAMPLE DATA SMOKE TESTS OK")


if __name__ == "__main__":
    run_all()
