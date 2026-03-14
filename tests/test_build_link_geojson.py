"""Tests for link GeoJSON normalization script."""

from __future__ import annotations


def test_build_features_with_mapping_row() -> None:
    from scripts.build_link_geojson import build_features

    centerlines = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"RAW_ID": "A-1", "SIG_CD": "11"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[126.97, 37.56], [126.98, 37.57]],
                },
            }
        ],
    }

    mapping_rows = {
        "A-1": {
            "LINK_ID": "seoul-link-001",
            "ROAD_NAME": "세종대로",
        }
    }

    features = build_features(
        centerlines=centerlines,
        source_name="seoul-gis",
        link_id_candidates=["RAW_ID"],
        road_name_candidates=["ROAD_NAME"],
        mapping_rows=mapping_rows,
        mapping_join_column="RAW_ID",
        mapping_link_id_column="LINK_ID",
        mapping_road_name_column="ROAD_NAME",
        region_property="SIG_CD",
        region_value="11",
    )

    assert len(features) == 1
    assert features[0]["properties"]["link_id"] == "seoul-link-001"
    assert features[0]["properties"]["road_name"] == "세종대로"
    assert features[0]["properties"]["source"] == "seoul-gis"
