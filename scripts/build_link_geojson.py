"""Build a normalized link GeoJSON for the local map matcher and map GUI.

Input:
  - GeoJSON FeatureCollection of road centerlines or link lines
  - Optional CSV mapping file with additional link metadata

Output:
  - GeoJSON FeatureCollection with properties:
      link_id, road_name, source

This script intentionally avoids heavy GIS dependencies so it can run in the
existing project environment. It expects the source geometry to already be
available as GeoJSON, for example:
  - exported from QGIS after loading official SHP/WFS data
  - downloaded directly from a GeoJSON-capable WFS endpoint
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_geojson(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Expected FeatureCollection: {path}")
    return data


def _load_mapping_csv(path: Path, key_column: str) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or key_column not in reader.fieldnames:
            raise ValueError(f"CSV key column '{key_column}' not found in {path}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            key = row.get(key_column)
            if key:
                rows[str(key)] = {k: (v or "") for k, v in row.items()}
        return rows


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_value(props: dict[str, Any], candidates: list[str]) -> str | None:
    for key in candidates:
        value = _coerce_str(props.get(key))
        if value is not None:
            return value
    return None


def _filter_seoul_feature(
    properties: dict[str, Any],
    *,
    region_property: str | None,
    region_value: str | None,
) -> bool:
    if not region_property or not region_value:
        return True
    value = _coerce_str(properties.get(region_property))
    return value == region_value


def build_features(
    *,
    centerlines: dict[str, Any],
    source_name: str,
    link_id_candidates: list[str],
    road_name_candidates: list[str],
    mapping_rows: dict[str, dict[str, str]] | None,
    mapping_join_column: str | None,
    mapping_link_id_column: str | None,
    mapping_road_name_column: str | None,
    region_property: str | None,
    region_value: str | None,
) -> list[dict[str, Any]]:
    normalized = []
    for feature in centerlines.get("features", []):
        properties = dict(feature.get("properties") or {})
        if not _filter_seoul_feature(
            properties,
            region_property=region_property,
            region_value=region_value,
        ):
            continue

        join_key = None
        if mapping_rows is not None and mapping_join_column is not None:
            join_key = _coerce_str(properties.get(mapping_join_column))
        mapping_row = (
            mapping_rows.get(join_key, {})
            if mapping_rows is not None and join_key else {}
        )

        merged_properties = {**properties, **mapping_row}
        link_id = _pick_value(
            merged_properties,
            ([mapping_link_id_column] if mapping_link_id_column else []) + link_id_candidates,
        )
        if link_id is None:
            continue

        road_name = _pick_value(
            merged_properties,
            ([mapping_road_name_column] if mapping_road_name_column else []) + road_name_candidates,
        )

        normalized.append(
            {
                "type": "Feature",
                "properties": {
                    "link_id": link_id,
                    "road_name": road_name,
                    "source": source_name,
                },
                "geometry": feature.get("geometry"),
            }
        )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized seoul_links.geojson")
    parser.add_argument("--centerlines", required=True, help="Input centerline/link GeoJSON")
    parser.add_argument("--output", default="data/gis/seoul_links.geojson")
    parser.add_argument("--source-name", default="seoul-gis")
    parser.add_argument(
        "--link-id-columns",
        default="link_id,LINK_ID,linkid,ID,id",
        help="Comma-separated property candidates for the output link_id",
    )
    parser.add_argument(
        "--road-name-columns",
        default="road_name,ROAD_NAME,도로명,rd_nm",
        help="Comma-separated property candidates for the output road_name",
    )
    parser.add_argument(
        "--mapping-csv", default=None, help="Optional metadata CSV to join",
    )
    parser.add_argument(
        "--mapping-join-column", default=None,
        help="Property name used to join CSV rows",
    )
    parser.add_argument(
        "--mapping-key-column", default=None, help="CSV key column",
    )
    parser.add_argument(
        "--mapping-link-id-column", default=None,
        help="CSV column for final link_id",
    )
    parser.add_argument("--mapping-road-name-column", default=None, help="CSV column for road_name")
    parser.add_argument(
        "--region-property",
        default=None,
        help="Optional property to filter, e.g. CTPRVN_NM or SIG_CD",
    )
    parser.add_argument(
        "--region-value",
        default=None,
        help="Expected region value, e.g. Seoul",
    )
    args = parser.parse_args()

    centerlines = _load_geojson(Path(args.centerlines))

    mapping_rows = None
    if args.mapping_csv:
        if args.mapping_key_column is None:
            raise ValueError("--mapping-key-column is required when --mapping-csv is used")
        mapping_rows = _load_mapping_csv(Path(args.mapping_csv), args.mapping_key_column)

    features = build_features(
        centerlines=centerlines,
        source_name=args.source_name,
        link_id_candidates=[c.strip() for c in args.link_id_columns.split(",") if c.strip()],
        road_name_candidates=[c.strip() for c in args.road_name_columns.split(",") if c.strip()],
        mapping_rows=mapping_rows,
        mapping_join_column=args.mapping_join_column,
        mapping_link_id_column=args.mapping_link_id_column,
        mapping_road_name_column=args.mapping_road_name_column,
        region_property=args.region_property,
        region_value=args.region_value,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            f,
            ensure_ascii=False,
        )

    print(f"Saved {len(features)} features to {output}")


if __name__ == "__main__":
    main()
