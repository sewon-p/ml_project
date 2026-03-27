"""Real-time ingestion endpoints: POST /ingest + POST /ingest-features.

Link-based mode (GIS available):
  1. POST /ingest receives GPS+accelerometer at 1Hz (bulk every 30s)
  2. SensorFusion (Kalman Filter) converts to FCD format
  3. LinkBuffer accumulates FCD across consecutive road links (1km target)
  4. On traversal completion → feature extraction → XGBoost inference
  5. Result registered with CF-weighted ensemble aggregator (15-min window)
  6. Prediction stored to all traversed links + WebSocket push

Legacy mode (no GIS):
  Falls back to 300s SessionBuffer → predict_density() → WebSocket push.

Also supports POST /ingest-features for client-side precomputed features.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from datetime import UTC, datetime
from functools import lru_cache
from typing import TypedDict, cast

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

_WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "300"))
_SESSION_TIMEOUT = 600  # 10 min
_LIVE_HISTORY_LIMIT = 20
_MATCH_SKIP_DISTANCE_M = 30.0  # skip GIS matching if moved less than this

# Link-based accumulation settings
_MIN_TRAVERSAL_DISTANCE_M = float(os.environ.get("MIN_TRAVERSAL_DISTANCE_M", "1000"))
_TARGET_TRAVERSAL_DISTANCE_M = float(os.environ.get("TARGET_TRAVERSAL_DISTANCE_M", "1000"))
_MIN_TRAVERSAL_RECORDS = int(os.environ.get("MIN_TRAVERSAL_RECORDS", "30"))
_LINK_EXIT_TIMEOUT = int(os.environ.get("LINK_EXIT_TIMEOUT", "120"))
_STICKY_LINK_COUNT = 1  # consecutive matches to new link before switching


@lru_cache(maxsize=1)
def _get_link_matcher():
    """Lazily load the local GIS matcher from config, if configured."""
    from src.gis import LinkMatcher

    config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
    matcher = LinkMatcher.from_config(config_path)
    if matcher is None:
        logger.info("GIS link matcher unavailable — ingest will run without map matching")
    else:
        logger.info("GIS link matcher loaded from %s", matcher.road_links_path)
    return matcher


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IngestRecord(BaseModel):
    """Single sensor reading from a mobile device."""

    session_id: str = Field(description="Unique session identifier")
    lat: float = Field(default=0.0, description="GPS latitude (degrees)")
    lon: float = Field(default=0.0, description="GPS longitude (degrees)")
    speed: float = Field(default=0.0, description="GPS speed (m/s)")
    heading: float = Field(default=0.0, description="GPS heading (degrees clockwise from north)")
    accel_x: float = Field(default=0.0, description="Longitudinal acceleration (m/s²)")
    accel_y: float = Field(default=0.0, description="Lateral acceleration (m/s²)")
    accel_z: float = Field(default=0.0, description="Vertical acceleration (m/s²)")
    timestamp: float = Field(default=0.0, description="Unix timestamp (seconds)")
    speed_limit: float = Field(default=22.22, description="Road speed limit (m/s)")
    num_lanes: int = Field(default=2, description="Number of lanes")
    link_id: str | None = Field(default=None, description="Matched road link ID")
    road_name: str | None = Field(default=None, description="Matched road name")
    geometry_geojson: str | None = Field(default=None, description="Matched link GeoJSON")
    center_lat: float | None = Field(default=None, description="Matched link center latitude")
    center_lon: float | None = Field(default=None, description="Matched link center longitude")
    link_source: str = Field(default="unknown", description="GIS source for the matched link")


class IngestResponse(BaseModel):
    """Response for POST /ingest."""

    status: str = "ok"
    session_id: str = ""
    buffer_count: int = 0
    prediction: dict | None = None
    traversal_completed: bool = False
    link_ids: list[str] | None = None
    traversal_distance_m: float | None = None
    ensemble_density: float | None = None
    ensemble_probe_count: int | None = None


class FeatureIngestRecord(BaseModel):
    """Client-side fused feature window for lightweight inference."""

    session_id: str = Field(description="Unique session identifier")
    timestamp: float = Field(default=0.0, description="Unix timestamp (seconds)")
    speed_limit: float = Field(default=22.22, description="Road speed limit (m/s)")
    num_lanes: int = Field(default=2, description="Number of lanes")
    buffer_count: int = Field(default=_WINDOW_SIZE, description="Window size used on device")
    features: dict[str, float] = Field(description="Client-computed scalar feature vector")
    lat: float | None = Field(default=None, description="Final GPS latitude (degrees)")
    lon: float | None = Field(default=None, description="Final GPS longitude (degrees)")
    heading: float | None = Field(
        default=None,
        description="Final GPS heading (degrees clockwise from north)",
    )


class LinkMeta(TypedDict, total=False):
    link_id: str
    road_name: str | None
    geometry_geojson: str | None
    center_lat: float | None
    center_lon: float | None
    source: str
    road_rank: str | None
    link_length_m: float | None
    lanes: int | None
    max_spd: float | None


class LinkTraversal(TypedDict):
    """Completed traversal of one or more consecutive links."""

    link_ids: list[str]
    link_metas: list[LinkMeta]
    total_distance_m: float
    traversal_time: float
    fcd_records: list[dict]
    speed_limit: float
    num_lanes: int


class LivePredictionEntry(TypedDict):
    prediction_id: int
    session_id: str
    observed_at: datetime
    density: float
    flow: float
    fd_density: float
    fd_flow: float
    residual_density: float


class LiveLinkState(TypedDict):
    link: LinkMeta
    history: list[LivePredictionEntry]


# ---------------------------------------------------------------------------
# In-process session buffer + fusion
# ---------------------------------------------------------------------------


class SessionBuffer:
    """Per-session FCD buffer with sensor fusion."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.buffer: deque[dict] = deque(maxlen=_WINDOW_SIZE)
        self.last_link: LinkMeta | None = None
        self.last_active = time.time()
        self.prediction_count = 0
        self.speed_limit = 22.22
        self.num_lanes = 2
        self._last_match_lat: float | None = None
        self._last_match_lon: float | None = None

        from src.streaming.fusion import SensorFusion

        self.fusion = SensorFusion(use_kalman=True)

    def add_raw(self, record: IngestRecord) -> tuple[dict, LinkMeta | None]:
        """Fuse a raw sensor reading and add to buffer. Returns FCD dict."""
        self.speed_limit = record.speed_limit
        self.num_lanes = record.num_lanes
        self.last_active = time.time()

        fcd = self.fusion.process(
            lat=record.lat,
            lon=record.lon,
            speed=record.speed,
            heading=record.heading,
            accel_x=record.accel_x,
            accel_y=record.accel_y,
            accel_z=record.accel_z,
            timestamp=record.timestamp,
        )
        self.buffer.append(fcd)

        matched_link: LinkMeta | None = None
        if record.link_id is not None:
            matched_link = {
                "link_id": record.link_id,
                "road_name": record.road_name,
                "geometry_geojson": record.geometry_geojson,
                "center_lat": record.center_lat,
                "center_lon": record.center_lon,
                "source": record.link_source,
                "road_rank": None,
                "link_length_m": None,
                "lanes": None,
                "max_spd": None,
            }
            self._last_match_lat = record.lat
            self._last_match_lon = record.lon
        elif self._should_rematch(record.lat, record.lon):
            matcher = _get_link_matcher()
            if matcher is not None:
                match = matcher.match(lat=record.lat, lon=record.lon, heading=record.heading)
                if match is not None:
                    matched_link = {
                        "link_id": match.link_id,
                        "road_name": match.road_name,
                        "geometry_geojson": match.geometry_geojson,
                        "center_lat": match.center_lat,
                        "center_lon": match.center_lon,
                        "source": match.source,
                        "road_rank": match.road_rank,
                        "link_length_m": match.link_length_m,
                        "lanes": match.lanes,
                        "max_spd": match.max_spd,
                    }
            self._last_match_lat = record.lat
            self._last_match_lon = record.lon
        else:
            # Reuse last match — position hasn't changed enough
            matched_link = self.last_link.copy() if self.last_link else None
        if matched_link is not None:
            self.last_link = cast(LinkMeta | None, matched_link)
        return fcd, matched_link

    def _should_rematch(self, lat: float, lon: float) -> bool:
        """Only re-run GIS matching if moved more than _MATCH_SKIP_DISTANCE_M."""
        if self._last_match_lat is None or self._last_match_lon is None:
            return True
        import math

        dlat = math.radians(lat - self._last_match_lat)
        dlon = math.radians(lon - self._last_match_lon) * math.cos(math.radians(lat))
        dist_m = math.sqrt(dlat * dlat + dlon * dlon) * 6_371_000.0
        return dist_m > _MATCH_SKIP_DISTANCE_M

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= _WINDOW_SIZE

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > _SESSION_TIMEOUT

    def get_window(self) -> list[dict]:
        return list(self.buffer)

    def get_representative_link(self) -> LinkMeta | None:
        return self.last_link.copy() if self.last_link is not None else None


class LinkBuffer:
    """Per-session link-based FCD accumulator.

    Accumulates FCD records as the probe traverses consecutive links.
    Triggers a traversal completion when accumulated distance reaches
    the target (default 700m). Stores FCD per-link for later ensemble.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.current_link_id: str | None = None
        self.current_link_meta: LinkMeta | None = None
        self.accumulated_links: list[LinkMeta] = []
        self.accumulated_distance_m: float = 0.0
        self.accumulated_records: list[dict] = []
        self.traversal_start_time: float = 0.0
        self.last_active: float = time.time()
        self.prediction_count: int = 0
        self.speed_limit: float = 22.22
        self.num_lanes: int = 2
        self._sticky_counter: int = 0
        self._sticky_candidate: str | None = None
        self._last_match_lat: float | None = None
        self._last_match_lon: float | None = None

        from src.streaming.fusion import SensorFusion

        self.fusion = SensorFusion(use_kalman=True)

    def add_raw(self, record: IngestRecord) -> tuple[dict, LinkMeta | None, LinkTraversal | None]:
        """Fuse sensor reading and accumulate. Returns completed traversal if ready."""
        self.speed_limit = record.speed_limit
        self.num_lanes = record.num_lanes
        self.last_active = time.time()

        fcd = self.fusion.process(
            lat=record.lat,
            lon=record.lon,
            speed=record.speed,
            heading=record.heading,
            accel_x=record.accel_x,
            accel_y=record.accel_y,
            accel_z=record.accel_z,
            timestamp=record.timestamp,
        )

        # Match to link
        matched_link = self._match_link(record)
        completed: LinkTraversal | None = None

        if matched_link is not None:
            new_link_id = matched_link.get("link_id")

            if self.current_link_id is None:
                # First link — start accumulating
                self._start_link(matched_link, fcd)
            elif new_link_id != self.current_link_id:
                # Sticky link: require N consecutive matches before switching
                if new_link_id == self._sticky_candidate:
                    self._sticky_counter += 1
                else:
                    self._sticky_candidate = new_link_id
                    self._sticky_counter = 1

                if self._sticky_counter >= _STICKY_LINK_COUNT:
                    # Confirmed link change
                    self._sticky_counter = 0
                    self._sticky_candidate = None

                    # Check if accumulated distance is enough
                    if self.accumulated_distance_m >= _MIN_TRAVERSAL_DISTANCE_M:
                        completed = self._complete_traversal()

                    # If not enough or just completed, start new accumulation
                    target_reached = self.accumulated_distance_m >= _TARGET_TRAVERSAL_DISTANCE_M
                    if completed is not None or target_reached:
                        self._start_link(matched_link, fcd)
                    else:
                        # Continue accumulating across links
                        self._add_link(matched_link, fcd)
                else:
                    # Not confirmed yet — keep recording on current link
                    self.accumulated_records.append(fcd)
            else:
                # Same link — just accumulate
                self.accumulated_records.append(fcd)
                self._sticky_counter = 0
                self._sticky_candidate = None
        else:
            # No match — still accumulate FCD
            self.accumulated_records.append(fcd)

        # Check if target distance reached
        if completed is None and self.accumulated_distance_m >= _TARGET_TRAVERSAL_DISTANCE_M:
            if len(self.accumulated_records) >= _MIN_TRAVERSAL_RECORDS:
                completed = self._complete_traversal()

        return fcd, matched_link, completed

    def _match_link(self, record: IngestRecord) -> LinkMeta | None:
        """Match GPS to road link."""
        if record.link_id is not None:
            return {
                "link_id": record.link_id,
                "road_name": record.road_name,
                "geometry_geojson": record.geometry_geojson,
                "center_lat": record.center_lat,
                "center_lon": record.center_lon,
                "source": record.link_source,
                "road_rank": None,
                "link_length_m": None,
                "lanes": None,
                "max_spd": None,
            }

        if not self._should_rematch(record.lat, record.lon):
            if self.current_link_meta is not None:
                return cast(LinkMeta, dict(self.current_link_meta))
            return None

        self._last_match_lat = record.lat
        self._last_match_lon = record.lon
        matcher = _get_link_matcher()
        if matcher is None:
            return None
        match = matcher.match(lat=record.lat, lon=record.lon, heading=record.heading)
        if match is None:
            return None
        return {
            "link_id": match.link_id,
            "road_name": match.road_name,
            "geometry_geojson": match.geometry_geojson,
            "center_lat": match.center_lat,
            "center_lon": match.center_lon,
            "source": match.source,
            "road_rank": match.road_rank,
            "link_length_m": match.link_length_m,
            "lanes": match.lanes,
            "max_spd": match.max_spd,
        }

    def _should_rematch(self, lat: float, lon: float) -> bool:
        if self._last_match_lat is None or self._last_match_lon is None:
            return True
        import math

        dlat = math.radians(lat - self._last_match_lat)
        dlon = math.radians(lon - self._last_match_lon) * math.cos(math.radians(lat))
        dist_m = math.sqrt(dlat * dlat + dlon * dlon) * 6_371_000.0
        return dist_m > _MATCH_SKIP_DISTANCE_M

    def _start_link(self, link_meta: LinkMeta, fcd: dict) -> None:
        """Start fresh accumulation on a new link."""
        self.current_link_id = link_meta.get("link_id")
        self.current_link_meta = link_meta
        self.accumulated_links = [link_meta]
        self.accumulated_distance_m = link_meta.get("link_length_m") or 0.0
        self.accumulated_records = [fcd]
        self.traversal_start_time = time.time()

    def _add_link(self, link_meta: LinkMeta, fcd: dict) -> None:
        """Add a new link to the current accumulation."""
        self.current_link_id = link_meta.get("link_id")
        self.current_link_meta = link_meta
        self.accumulated_links.append(link_meta)
        self.accumulated_distance_m += link_meta.get("link_length_m") or 0.0
        self.accumulated_records.append(fcd)

    def _complete_traversal(self) -> LinkTraversal:
        """Package accumulated data into a completed traversal."""
        traversal: LinkTraversal = {
            "link_ids": [lm.get("link_id", "") for lm in self.accumulated_links],
            "link_metas": list(self.accumulated_links),
            "total_distance_m": self.accumulated_distance_m,
            "traversal_time": time.time() - self.traversal_start_time,
            "fcd_records": list(self.accumulated_records),
            "speed_limit": self.speed_limit,
            "num_lanes": self.num_lanes,
        }
        self.prediction_count += 1
        # Reset accumulation
        self.accumulated_links = []
        self.accumulated_distance_m = 0.0
        self.accumulated_records = []
        self.current_link_id = None
        return traversal

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > _SESSION_TIMEOUT


class SessionManager:
    """Manages per-session buffers (legacy + link-based) and runs predictions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionBuffer] = {}
        self._link_sessions: dict[str, LinkBuffer] = {}

    def get_or_create(self, session_id: str) -> SessionBuffer:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBuffer(session_id)
            logger.info("New session: %s", session_id)
        return self._sessions[session_id]

    def get_or_create_link(self, session_id: str) -> LinkBuffer:
        if session_id not in self._link_sessions:
            self._link_sessions[session_id] = LinkBuffer(session_id)
            logger.info("New link session: %s", session_id)
        return self._link_sessions[session_id]

    def cleanup_expired(self) -> None:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]
            logger.info("Expired session: %s", sid)
        expired_link = [sid for sid, s in self._link_sessions.items() if s.is_expired]
        for sid in expired_link:
            del self._link_sessions[sid]
            logger.info("Expired link session: %s", sid)

    def predict(self, session: SessionBuffer, registry: object) -> dict | None:
        """Run XGBoost prediction on a session's 300s window (legacy)."""
        if not session.ready:
            return None

        from src.api.inference import predict_density

        try:
            result = predict_density(
                fcd_records=session.get_window(),
                speed_limit=session.speed_limit,
                num_lanes=session.num_lanes,
                registry=registry,  # type: ignore[arg-type]
            )
            session.prediction_count += 1
            return {
                "session_id": session.session_id,
                "timestamp": time.time(),
                **result,
            }
        except Exception:
            logger.warning(
                "Prediction failed for session %s",
                session.session_id,
                exc_info=True,
            )
            return None

    def predict_traversal(
        self, traversal: LinkTraversal, session_id: str, registry: object
    ) -> dict | None:
        """Run XGBoost on a completed link traversal."""
        from src.api.inference import predict_density_from_traversal

        try:
            result = predict_density_from_traversal(
                traversal=dict(traversal),  # type: ignore[arg-type]
                registry=registry,  # type: ignore[arg-type]
            )
            return {
                "session_id": session_id,
                "timestamp": time.time(),
                **result,
            }
        except Exception:
            logger.warning(
                "Traversal prediction failed for %s",
                session_id,
                exc_info=True,
            )
            return None


sessions = SessionManager()
_ensemble_aggregator: object | None = None


def _get_ensemble_aggregator():
    """Lazily create the ensemble aggregator."""
    global _ensemble_aggregator
    if _ensemble_aggregator is None:
        from src.api.ensemble import EnsembleAggregator

        _ensemble_aggregator = EnsembleAggregator(window_seconds=900.0, temperature=1.0)
        logger.info("Ensemble aggregator initialized (15-min window)")
    return _ensemble_aggregator


_live_link_history: dict[str, LiveLinkState] = {}
_live_prediction_index: dict[int, tuple[str, LivePredictionEntry]] = {}


def _store_live_prediction(link_meta: LinkMeta, prediction: dict, session_id: str) -> None:
    """Keep recent link predictions in memory so the map can update without a DB."""
    link_id = str(link_meta["link_id"])
    observed_at = datetime.fromtimestamp(prediction["timestamp"], tz=UTC)
    raw_prediction_id = prediction.get("prediction_id")
    prediction_id = (
        int(cast(int | float | str, raw_prediction_id))
        if raw_prediction_id is not None
        else int(float(prediction["timestamp"]) * 1000)
    )
    entry: LivePredictionEntry = {
        "prediction_id": int(prediction_id),
        "session_id": session_id,
        "observed_at": observed_at,
        "density": float(prediction["density"]),
        "flow": float(prediction["flow"]),
        "fd_density": float(prediction["fd_density"]),
        "fd_flow": float(prediction["fd_flow"]),
        "residual_density": float(prediction["residual_density"]),
    }

    state = _live_link_history.setdefault(
        link_id,
        {
            "link": {
                "link_id": link_id,
                "road_name": link_meta.get("road_name"),
                "source": link_meta.get("source", "live"),
                "geometry_geojson": link_meta.get("geometry_geojson"),
                "center_lat": link_meta.get("center_lat"),
                "center_lon": link_meta.get("center_lon"),
            },
            "history": [],
        },
    )
    history = state["history"]
    history.insert(0, entry)
    del history[_LIVE_HISTORY_LIMIT:]
    _live_prediction_index[entry["prediction_id"]] = (link_id, entry)


def _match_last_link(
    *,
    lat: float | None,
    lon: float | None,
    heading: float | None = None,
) -> LinkMeta | None:
    if lat is None or lon is None:
        return None
    matcher = _get_link_matcher()
    if matcher is None:
        return None
    match = matcher.match(lat=lat, lon=lon)
    if match is None:
        return None
    return {
        "link_id": match.link_id,
        "road_name": match.road_name,
        "geometry_geojson": match.geometry_geojson,
        "center_lat": match.center_lat,
        "center_lon": match.center_lon,
        "source": match.source,
    }


def get_live_map_snapshot() -> dict[str, LiveLinkState]:
    """Return in-memory link predictions generated through /ingest."""
    return _live_link_history


def get_live_prediction_detail(
    prediction_id: int,
) -> tuple[LinkMeta, LivePredictionEntry] | None:
    """Return one in-memory prediction and its link metadata."""
    found = _live_prediction_index.get(prediction_id)
    if found is None:
        return None
    link_id, entry = found
    state = _live_link_history.get(link_id)
    if state is None:
        return None
    return state["link"], entry


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages WebSocket connections grouped by session_id."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = []
        self._connections[session_id].append(websocket)
        logger.info("WebSocket connected: session=%s", session_id)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        if session_id in self._connections:
            self._connections[session_id] = [
                ws for ws in self._connections[session_id] if ws is not websocket
            ]
            if not self._connections[session_id]:
                del self._connections[session_id]
        logger.info("WebSocket disconnected: session=%s", session_id)

    async def send_to_session(self, session_id: str, data: dict) -> None:
        """Send data to all WebSocket clients for a given session."""
        if session_id not in self._connections:
            return
        stale: list[WebSocket] = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(data)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.disconnect(session_id, ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# POST /ingest — data collection + in-process prediction
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest(record: IngestRecord, request: Request) -> IngestResponse:
    """Receive GPS+accelerometer reading, fuse, accumulate, and predict.

    Two modes:
      - Link-based (GIS available): accumulate across links until 700m,
        then predict and register with ensemble aggregator.
      - Legacy (no GIS): 300s sliding window, predict when full.
    """
    registry = getattr(request.app.state, "registry", None)
    use_link_mode = _get_link_matcher() is not None

    # ── Link-based mode ──────────────────────────────────────
    if use_link_mode:
        link_session = sessions.get_or_create_link(record.session_id)

        # Auto-fill speed_limit/num_lanes from link match
        matcher = _get_link_matcher()
        if matcher and record.lat and record.lon:
            match = matcher.match(lat=record.lat, lon=record.lon, heading=record.heading)
            if match and match.max_spd:
                record.speed_limit = match.max_spd / 3.6  # km/h → m/s
            if match and match.lanes:
                record.num_lanes = match.lanes

        fcd, matched_link, traversal = link_session.add_raw(record)

        prediction = None
        ensemble_density = None
        ensemble_count = None
        completed_link_ids = None

        if traversal is not None and registry is not None:
            prediction = sessions.predict_traversal(traversal, record.session_id, registry)

            if prediction:
                completed_link_ids = traversal["link_ids"]
                cf_score = prediction.get("cf_score", 0.0)

                # Register with ensemble for each link
                aggregator = _get_ensemble_aggregator()
                for link_meta in traversal["link_metas"]:
                    lid = link_meta.get("link_id", "")
                    if not lid:
                        continue
                    cf_feats = prediction.get("cf_features", {})
                    ens = aggregator.add_prediction(
                        link_id=lid,
                        density=prediction["density"],
                        flow=prediction["flow"],
                        features=cf_feats,
                        prediction_id=prediction.get("prediction_id"),
                        session_id=record.session_id,
                    )
                    ensemble_density = ens.ensemble_density
                    ensemble_count = ens.probe_count

                    # Store live for map
                    _store_live_prediction(cast(LinkMeta, link_meta), prediction, record.session_id)

                # DB save (best-effort)
                if getattr(request.app.state, "db_available", False):
                    try:
                        from src.api.crud import save_prediction
                        from src.api.database import async_session_factory

                        assert async_session_factory is not None
                        rep = traversal["link_metas"][0] if traversal["link_metas"] else {}
                        async with async_session_factory() as db_session:
                            pid = await save_prediction(
                                session=db_session,
                                speed_limit=traversal["speed_limit"],
                                num_lanes=traversal["num_lanes"],
                                fcd_records=traversal["fcd_records"],
                                result=prediction,
                                session_id=record.session_id,
                                link_id=rep.get("link_id"),
                                road_name=rep.get("road_name"),
                                geometry_geojson=rep.get("geometry_geojson"),
                                center_lat=rep.get("center_lat"),
                                center_lon=rep.get("center_lon"),
                                source=str(rep.get("source", "live")),
                                link_length_m=traversal["total_distance_m"],
                                traversal_time=traversal["traversal_time"],
                                cf_weight=cf_score,
                                observed_at=datetime.fromtimestamp(record.timestamp, tz=UTC),
                            )
                        prediction["prediction_id"] = pid
                    except Exception:
                        logger.warning("DB save failed", exc_info=True)

                await manager.send_to_session(record.session_id, prediction)

        # Periodic maintenance
        sessions.cleanup_expired()
        aggregator = _get_ensemble_aggregator()
        aggregator.freeze_stale()
        aggregator.cleanup(max_age_seconds=3600)

        return IngestResponse(
            status="ok",
            session_id=record.session_id,
            buffer_count=len(link_session.accumulated_records),
            prediction=prediction,
            traversal_completed=traversal is not None,
            link_ids=completed_link_ids,
            traversal_distance_m=(traversal["total_distance_m"] if traversal else None),
            ensemble_density=ensemble_density,
            ensemble_probe_count=ensemble_count,
        )

    # ── Legacy 300s mode (no GIS) ────────────────────────────
    session = sessions.get_or_create(record.session_id)
    _, matched_link = session.add_raw(record)

    prediction = None
    if session.ready and registry is not None:
        prediction = sessions.predict(session, registry)

        if prediction:
            representative_link = session.get_representative_link()
            if getattr(request.app.state, "db_available", False):
                try:
                    from src.api.crud import save_prediction
                    from src.api.database import async_session_factory

                    assert async_session_factory is not None
                    async with async_session_factory() as db_session:
                        prediction_id = await save_prediction(
                            session=db_session,
                            speed_limit=record.speed_limit,
                            num_lanes=record.num_lanes,
                            fcd_records=session.get_window(),
                            result=prediction,
                            session_id=record.session_id,
                            link_id=(representative_link or {}).get("link_id"),
                            road_name=(representative_link or {}).get("road_name"),
                            geometry_geojson=(representative_link or {}).get("geometry_geojson"),
                            center_lat=(representative_link or {}).get("center_lat"),
                            center_lon=(representative_link or {}).get("center_lon"),
                            source=str((representative_link or {}).get("source", "unknown")),
                            observed_at=datetime.fromtimestamp(record.timestamp, tz=UTC),
                        )
                    prediction["prediction_id"] = prediction_id
                except Exception:
                    logger.warning("DB save failed", exc_info=True)
            if representative_link is not None:
                prediction["link_id"] = representative_link["link_id"]
                prediction["road_name"] = representative_link.get("road_name")
                _store_live_prediction(
                    cast(LinkMeta, representative_link),
                    prediction,
                    record.session_id,
                )
            elif matched_link is not None:
                prediction["link_id"] = matched_link["link_id"]
                prediction["road_name"] = matched_link.get("road_name")
                _store_live_prediction(cast(LinkMeta, matched_link), prediction, record.session_id)
            await manager.send_to_session(record.session_id, prediction)

    try:
        from src.streaming.producer import TOPIC_FCD_RAW, publish

        publish(TOPIC_FCD_RAW, key=record.session_id, value=record.model_dump())
    except Exception:
        pass

    sessions.cleanup_expired()

    return IngestResponse(
        status="ok",
        session_id=record.session_id,
        buffer_count=len(session.buffer),
        prediction=prediction,
    )


@router.post("/ingest-features", response_model=IngestResponse)
async def ingest_features(record: FeatureIngestRecord, request: Request) -> IngestResponse:
    """Receive a client-computed feature window and run lightweight inference."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return IngestResponse(
            status="error",
            session_id=record.session_id,
            buffer_count=record.buffer_count,
            prediction=None,
        )

    from src.api.inference import predict_density_from_features

    base_prediction = predict_density_from_features(
        features=record.features,
        speed_limit=record.speed_limit,
        num_lanes=record.num_lanes,
        registry=registry,  # type: ignore[arg-type]
    )
    prediction: dict[str, object] = {
        "session_id": record.session_id,
        "timestamp": record.timestamp or time.time(),
        **base_prediction,
    }

    matched_link = _match_last_link(lat=record.lat, lon=record.lon, heading=record.heading)
    if getattr(request.app.state, "db_available", False):
        try:
            from datetime import UTC, datetime

            from src.api.crud import save_prediction
            from src.api.database import async_session_factory

            assert async_session_factory is not None
            async with async_session_factory() as db_session:
                prediction_id = await save_prediction(
                    session=db_session,
                    speed_limit=record.speed_limit,
                    num_lanes=record.num_lanes,
                    fcd_records=[],
                    result=prediction,
                    session_id=record.session_id,
                    link_id=(matched_link or {}).get("link_id"),
                    road_name=(matched_link or {}).get("road_name"),
                    geometry_geojson=(matched_link or {}).get("geometry_geojson"),
                    center_lat=(matched_link or {}).get("center_lat"),
                    center_lon=(matched_link or {}).get("center_lon"),
                    source=str((matched_link or {}).get("source", "unknown")),
                    observed_at=datetime.fromtimestamp(record.timestamp, tz=UTC),
                )
            prediction["prediction_id"] = prediction_id
        except Exception:
            logger.warning("Failed to save feature ingest prediction to DB", exc_info=True)

    if matched_link is not None:
        prediction["link_id"] = matched_link["link_id"]
        prediction["road_name"] = matched_link.get("road_name")
        _store_live_prediction(cast(LinkMeta, matched_link), prediction, record.session_id)

    await manager.send_to_session(record.session_id, prediction)

    return IngestResponse(
        status="ok",
        session_id=record.session_id,
        buffer_count=record.buffer_count,
        prediction=prediction,
    )


# ---------------------------------------------------------------------------
# WebSocket /ws/dashboard/{session_id}
# ---------------------------------------------------------------------------


@router.websocket("/ws/dashboard/{session_id}")
async def dashboard_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time prediction results.

    Predictions are pushed here automatically when POST /ingest
    accumulates 300s of data for this session.
    """
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
    except Exception:
        manager.disconnect(session_id, websocket)
