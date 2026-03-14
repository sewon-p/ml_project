"""Tests for local GIS link matching."""

from __future__ import annotations

import json


def test_link_matcher_finds_nearest_link(tmp_path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"link_id": "link-a", "road_name": "A Road"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [126.9700, 37.5600],
                        [126.9800, 37.5600],
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"link_id": "link-b", "road_name": "B Road"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [126.9700, 37.5700],
                        [126.9800, 37.5700],
                    ],
                },
            },
        ],
    }
    geojson_path = tmp_path / "links.geojson"
    geojson_path.write_text(json.dumps(geojson), encoding="utf-8")

    from src.gis import LinkMatcher

    matcher = LinkMatcher(geojson_path, source="test-gis", max_match_distance_m=100.0)
    match = matcher.match(lat=37.5601, lon=126.9750, heading=90.0)

    assert match is not None
    assert match.link_id == "link-a"
    assert match.road_name == "A Road"
    assert match.source == "test-gis"


def test_link_matcher_respects_max_distance(tmp_path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"link_id": "link-a"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [126.9700, 37.5600],
                        [126.9800, 37.5600],
                    ],
                },
            }
        ],
    }
    geojson_path = tmp_path / "links.geojson"
    geojson_path.write_text(json.dumps(geojson), encoding="utf-8")

    from src.gis import LinkMatcher

    matcher = LinkMatcher(geojson_path, max_match_distance_m=10.0)
    match = matcher.match(lat=37.5900, lon=127.0200, heading=180.0)

    assert match is None
