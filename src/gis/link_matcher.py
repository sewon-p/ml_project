"""Nearest-link matcher for mobile GPS points against local GeoJSON road links."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_config

_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class RoadLinkMatch:
    """Matched link metadata used for prediction storage and map display."""

    link_id: str
    road_name: str | None
    geometry_geojson: str
    center_lat: float
    center_lon: float
    source: str
    distance_m: float
    road_rank: str | None = None
    link_length_m: float | None = None
    lanes: int | None = None
    max_spd: float | None = None


@dataclass(frozen=True)
class _PreparedLink:
    link_id: str
    road_name: str | None
    geometry_geojson: str
    source: str
    center_lat: float
    center_lon: float
    segments: list[tuple[tuple[float, float], tuple[float, float]]]
    road_rank: str | None = None
    link_length_m: float | None = None
    lanes: int | None = None
    max_spd: float | None = None


def _heading_delta_deg(a: float, b: float) -> float:
    delta = abs((a - b) % 360.0)
    return min(delta, 360.0 - delta)


def _project_to_local_m(
    lat: float, lon: float, ref_lat: float, ref_lon: float
) -> tuple[float, float]:
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    x = dlon * _EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    y = dlat * _EARTH_RADIUS_M
    return x, y


def _segment_distance_m(
    point_xy: tuple[float, float],
    seg_start_xy: tuple[float, float],
    seg_end_xy: tuple[float, float],
) -> tuple[float, float]:
    px, py = point_xy
    x1, y1 = seg_start_xy
    x2, y2 = seg_end_xy
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - x1, py - y1), 0.0

    t = ((px - x1) * dx + (py - y1) * dy) / denom
    t = min(1.0, max(0.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    distance = math.hypot(px - proj_x, py - proj_y)
    heading = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return distance, heading


def _flatten_lines(geometry: dict[str, Any]) -> list[list[tuple[float, float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "LineString":
        return [[(float(lon), float(lat)) for lon, lat in coordinates]]
    if geometry_type == "MultiLineString":
        return [[(float(lon), float(lat)) for lon, lat in line] for line in coordinates]
    return []


_GRID_CELL_DEG = 0.001  # ~100m at mid-latitudes


class LinkMatcher:
    """Loads GeoJSON road links and matches GPS points to the nearest link.

    Uses a grid-based spatial index (cell size ~100 m) so that each ``match()``
    call only examines the handful of links near the query point instead of
    scanning all 400 k+ links.
    """

    def __init__(
        self,
        road_links_path: str | Path,
        *,
        source: str = "unknown",
        link_id_property: str = "link_id",
        road_name_property: str = "road_name",
        max_match_distance_m: float = 40.0,
        heading_weight_m: float = 15.0,
    ) -> None:
        self.road_links_path = Path(road_links_path)
        self.source = source
        self.link_id_property = link_id_property
        self.road_name_property = road_name_property
        self.max_match_distance_m = max_match_distance_m
        self.heading_weight_m = heading_weight_m
        self.min_road_rank: int | None = None
        self.min_link_length_m: float | None = None
        self.links = self._load_links()
        self._grid: dict[tuple[int, int], list[_PreparedLink]] = {}
        self._build_grid()

    @classmethod
    def from_config(cls, config_path: str | Path) -> LinkMatcher | None:
        cfg = load_config(config_path)
        gis_cfg = cfg.get("gis", {})
        if not gis_cfg.get("enabled", False):
            return None

        road_links_path = gis_cfg.get("road_links_path")
        if not road_links_path:
            return None

        path = Path(road_links_path)
        if not path.is_absolute():
            path = (Path(config_path).resolve().parent.parent / path).resolve()
        if not path.exists():
            return None

        matcher = cls(
            road_links_path=path,
            source=gis_cfg.get("source", "unknown"),
            link_id_property=gis_cfg.get("link_id_property", "link_id"),
            road_name_property=gis_cfg.get("road_name_property", "road_name"),
            max_match_distance_m=float(gis_cfg.get("max_match_distance_m", 40.0)),
            heading_weight_m=float(gis_cfg.get("heading_weight_m", 15.0)),
        )
        if "min_road_rank" in gis_cfg:
            matcher.min_road_rank = int(gis_cfg["min_road_rank"])
        if "min_link_length_m" in gis_cfg:
            matcher.min_link_length_m = float(gis_cfg["min_link_length_m"])
        return matcher

    def _load_links(self) -> list[_PreparedLink]:
        with open(self.road_links_path, encoding="utf-8") as f:
            geojson = json.load(f)

        features = geojson.get("features", [])
        prepared: list[_PreparedLink] = []
        for feature in features:
            geometry = feature.get("geometry") or {}
            lines = _flatten_lines(geometry)
            if not lines:
                continue

            props = feature.get("properties") or {}
            link_id = props.get(self.link_id_property) or props.get("id")
            if link_id is None:
                continue

            all_points = [pt for line in lines for pt in line]
            center_lon = sum(pt[0] for pt in all_points) / len(all_points)
            center_lat = sum(pt[1] for pt in all_points) / len(all_points)
            segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
            for line in lines:
                for idx in range(len(line) - 1):
                    segments.append((line[idx], line[idx + 1]))
            if not segments:
                continue

            road_rank = props.get("road_rank")
            link_length_m = props.get("link_length_m")
            lanes = props.get("lanes")
            max_spd = props.get("max_spd")

            # Filter by road rank (lower = higher hierarchy)
            if self.min_road_rank is not None and road_rank is not None:
                try:
                    if int(road_rank) > self.min_road_rank:
                        continue
                except (ValueError, TypeError):
                    pass

            # Filter by minimum link length
            if self.min_link_length_m is not None and link_length_m is not None:
                try:
                    if float(link_length_m) < self.min_link_length_m:
                        continue
                except (ValueError, TypeError):
                    pass

            prepared.append(
                _PreparedLink(
                    link_id=str(link_id),
                    road_name=props.get(self.road_name_property),
                    geometry_geojson=json.dumps(geometry, ensure_ascii=False),
                    source=self.source,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    segments=segments,
                    road_rank=str(road_rank) if road_rank is not None else None,
                    link_length_m=float(link_length_m) if link_length_m is not None else None,
                    lanes=int(lanes) if lanes is not None else None,
                    max_spd=float(max_spd) if max_spd is not None else None,
                )
            )
        return prepared

    # ------------------------------------------------------------------
    # Grid spatial index
    # ------------------------------------------------------------------

    @staticmethod
    def _cell(lat: float, lon: float) -> tuple[int, int]:
        return int(math.floor(lat / _GRID_CELL_DEG)), int(math.floor(lon / _GRID_CELL_DEG))

    def _build_grid(self) -> None:
        """Assign each link to every grid cell its segments touch."""
        for link in self.links:
            cells_seen: set[tuple[int, int]] = set()
            # Index by center
            cells_seen.add(self._cell(link.center_lat, link.center_lon))
            # Index by each segment endpoint
            for (lon1, lat1), (lon2, lat2) in link.segments:
                cells_seen.add(self._cell(lat1, lon1))
                cells_seen.add(self._cell(lat2, lon2))
            for cell in cells_seen:
                self._grid.setdefault(cell, []).append(link)

    def _nearby_links(self, lat: float, lon: float) -> list[_PreparedLink]:
        """Return links in the query cell and its 8 neighbours."""
        ci, cj = self._cell(lat, lon)
        candidates: list[_PreparedLink] = []
        seen_ids: set[str] = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for link in self._grid.get((ci + di, cj + dj), ()):
                    if link.link_id not in seen_ids:
                        seen_ids.add(link.link_id)
                        candidates.append(link)
        return candidates

    # ------------------------------------------------------------------
    # Match
    # ------------------------------------------------------------------

    def match(
        self,
        *,
        lat: float,
        lon: float,
        heading: float | None = None,
    ) -> RoadLinkMatch | None:
        if not self.links:
            return None

        candidates = self._nearby_links(lat, lon)
        if not candidates:
            return None

        best_link = None
        best_score = float("inf")
        best_distance = float("inf")

        for link in candidates:
            point_xy = _project_to_local_m(lat, lon, link.center_lat, link.center_lon)
            min_distance = float("inf")
            best_segment_heading = 0.0
            for (lon1, lat1), (lon2, lat2) in link.segments:
                start_xy = _project_to_local_m(lat1, lon1, link.center_lat, link.center_lon)
                end_xy = _project_to_local_m(lat2, lon2, link.center_lat, link.center_lon)
                distance, segment_heading = _segment_distance_m(point_xy, start_xy, end_xy)
                if distance < min_distance:
                    min_distance = distance
                    best_segment_heading = segment_heading

            score = min_distance
            if heading is not None:
                score += (
                    _heading_delta_deg(heading, best_segment_heading) / 180.0
                ) * self.heading_weight_m

            if score < best_score:
                best_score = score
                best_distance = min_distance
                best_link = link

        if best_link is None or best_distance > self.max_match_distance_m:
            return None

        return RoadLinkMatch(
            link_id=best_link.link_id,
            road_name=best_link.road_name,
            geometry_geojson=best_link.geometry_geojson,
            center_lat=best_link.center_lat,
            center_lon=best_link.center_lon,
            source=best_link.source,
            distance_m=best_distance,
            road_rank=best_link.road_rank,
            link_length_m=best_link.link_length_m,
            lanes=best_link.lanes,
            max_spd=best_link.max_spd,
        )
