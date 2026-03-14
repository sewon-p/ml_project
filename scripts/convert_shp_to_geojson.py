"""Convert the supplied Seoul road centerline shapefile to WGS84 GeoJSON.

This converter is dependency-free and supports the specific projection found in
`raw/Seoul_road_data/N3L_A0020000_11.prj`:
Korea 2000 / Unified CS (Transverse Mercator).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def _parse_dbf_records(path: Path) -> list[dict[str, str]]:
    with path.open("rb") as f:
        header = f.read(32)
        _, _, _, _, num_records, header_len, record_len = struct.unpack("<BBBBIHH20x", header)
        num_fields = (header_len - 33) // 32
        fields: list[tuple[str, int]] = []
        for _ in range(num_fields):
            desc = f.read(32)
            name = desc[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            length = desc[16]
            fields.append((name, length))
        f.read(1)  # field descriptor terminator

        rows: list[dict[str, str]] = []
        for _ in range(num_records):
            record = f.read(record_len)
            if not record or record[0] == 0x2A:
                continue
            pos = 1
            row: dict[str, str] = {}
            for name, length in fields:
                raw = record[pos : pos + length]
                pos += length
                row[name] = raw.decode("euc-kr", errors="ignore").strip()
            rows.append(row)
    return rows


def _parse_shp_polylines(path: Path) -> list[list[list[tuple[float, float]]]]:
    polylines: list[list[list[tuple[float, float]]]] = []
    with path.open("rb") as f:
        f.read(100)  # file header
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            _, content_length_words = struct.unpack(">2i", header)
            content = f.read(content_length_words * 2)
            if len(content) < 44:
                break
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                polylines.append([])
                continue
            if shape_type != 3:
                raise ValueError(f"Unsupported shape type: {shape_type}")

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts = struct.unpack(f"<{num_parts}i", content[44 : 44 + 4 * num_parts])
            off = 44 + 4 * num_parts
            points = [
                struct.unpack("<2d", content[off + i * 16 : off + (i + 1) * 16])
                for i in range(num_points)
            ]
            line_parts: list[list[tuple[float, float]]] = []
            for idx, start in enumerate(parts):
                end = parts[idx + 1] if idx + 1 < len(parts) else num_points
                line_parts.append(points[start:end])
            polylines.append(line_parts)
    return polylines


def _tm_to_wgs84(x: float, y: float) -> tuple[float, float]:
    a = 6378137.0
    f = 1 / 298.257222101
    e2 = 2 * f - f * f
    e_prime2 = e2 / (1 - e2)
    k0 = 0.9996
    lon0 = math.radians(127.5)
    lat0 = math.radians(38.0)
    false_easting = 1_000_000.0
    false_northing = 2_000_000.0

    def meridional_arc(phi: float) -> float:
        e4 = e2 * e2
        e6 = e4 * e2
        return a * (
            (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
            - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
            + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
            - (35 * e6 / 3072) * math.sin(6 * phi)
        )

    m0 = meridional_arc(lat0)
    m1 = m0 + (y - false_northing) / k0
    mu1 = m1 / (
        a
        * (
            1
            - e2 / 4
            - 3 * e2 * e2 / 64
            - 5 * e2 * e2 * e2 / 256
        )
    )

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu1 + j1 * math.sin(2 * mu1) + j2 * math.sin(4 * mu1) + j3 * math.sin(6 * mu1) + j4 * math.sin(8 * mu1)

    sin_fp = math.sin(fp)
    cos_fp = math.cos(fp)
    tan_fp = math.tan(fp)
    c1 = e_prime2 * cos_fp * cos_fp
    t1 = tan_fp * tan_fp
    n1 = a / math.sqrt(1 - e2 * sin_fp * sin_fp)
    r1 = a * (1 - e2) / (1 - e2 * sin_fp * sin_fp) ** 1.5
    d = (x - false_easting) / (n1 * k0)

    lat = fp - (
        n1
        * tan_fp
        / r1
        * (
            d * d / 2
            - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * e_prime2) * d**4 / 24
            + (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * e_prime2 - 3 * c1 * c1) * d**6 / 720
        )
    )
    lon = lon0 + (
        d
        - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * e_prime2 + 24 * t1 * t1) * d**5 / 120
    ) / cos_fp

    return math.degrees(lon), math.degrees(lat)


def convert(shp_path: Path, dbf_path: Path) -> dict:
    geometries = _parse_shp_polylines(shp_path)
    records = _parse_dbf_records(dbf_path)
    if len(geometries) != len(records):
        raise ValueError(f"Shape/DBF count mismatch: {len(geometries)} vs {len(records)}")

    features = []
    for geom_parts, record in zip(geometries, records):
        if not geom_parts:
            continue
        lines = []
        for part in geom_parts:
            lines.append([_tm_to_wgs84(x, y) for x, y in part])

        geometry = (
            {"type": "LineString", "coordinates": lines[0]}
            if len(lines) == 1
            else {"type": "MultiLineString", "coordinates": lines}
        )
        road_name = record.get("RDNM") or record.get("NAME") or None
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "link_id": record.get("UFID") or record.get("FMTA"),
                    "road_name": road_name,
                    "source": "seoul-road-centerline",
                },
                "geometry": geometry,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Seoul road centerline SHP to GeoJSON")
    parser.add_argument("--shp", default="raw/Seoul_road_data/N3L_A0020000_11.shp")
    parser.add_argument("--dbf", default="raw/Seoul_road_data/N3L_A0020000_11.dbf")
    parser.add_argument("--output", default="data/gis/seoul_links.geojson")
    args = parser.parse_args()

    geojson = convert(Path(args.shp), Path(args.dbf))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"Saved {len(geojson['features'])} features to {output}")


if __name__ == "__main__":
    main()
