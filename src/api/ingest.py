"""Real-time ingestion endpoints: POST /ingest + WebSocket /ws/dashboard.

Handles the full pipeline in-process:
  1. POST /ingest receives GPS+accelerometer data
  2. SensorFusion (Kalman Filter) converts to FCD format
  3. SessionBuffer accumulates 300s sliding window
  4. predict_density() runs XGBoost inference
  5. Result pushed to WebSocket clients in real-time

Also publishes to Kafka/Pub/Sub if available (for external consumers).
"""

from __future__ import annotations

import logging
import time
from collections import deque

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

_WINDOW_SIZE = 300
_SESSION_TIMEOUT = 600  # 10 min


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


class IngestResponse(BaseModel):
    """Response for POST /ingest."""

    status: str = "ok"
    session_id: str = ""
    buffer_count: int = 0
    prediction: dict | None = None


# ---------------------------------------------------------------------------
# In-process session buffer + fusion
# ---------------------------------------------------------------------------


class SessionBuffer:
    """Per-session FCD buffer with sensor fusion."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.buffer: deque[dict] = deque(maxlen=_WINDOW_SIZE)
        self.last_active = time.time()
        self.prediction_count = 0
        self.speed_limit = 22.22
        self.num_lanes = 2

        from src.streaming.fusion import SensorFusion

        self.fusion = SensorFusion(use_kalman=True)

    def add_raw(self, record: IngestRecord) -> dict:
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
        return fcd

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= _WINDOW_SIZE

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > _SESSION_TIMEOUT

    def get_window(self) -> list[dict]:
        return list(self.buffer)


class SessionManager:
    """Manages per-session buffers and runs predictions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionBuffer] = {}

    def get_or_create(self, session_id: str) -> SessionBuffer:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionBuffer(session_id)
            logger.info("New session: %s", session_id)
        return self._sessions[session_id]

    def cleanup_expired(self) -> None:
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]
            logger.info("Expired session: %s", sid)

    def predict(self, session: SessionBuffer, registry: object) -> dict | None:
        """Run XGBoost prediction on a session's 300s window."""
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


sessions = SessionManager()


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

    Flow:
      1. Kalman Filter sensor fusion → FCD record
      2. Add to session's 300s sliding window
      3. If 300s reached → run XGBoost predict_density()
      4. Push result to WebSocket clients
      5. Also publish to Kafka/Pub/Sub if available
    """
    # 1-2. Fuse and buffer
    session = sessions.get_or_create(record.session_id)
    session.add_raw(record)

    # 3. Predict if buffer is full
    prediction = None
    if session.ready:
        registry = getattr(request.app.state, "registry", None)
        if registry is not None:
            prediction = sessions.predict(session, registry)

            # 4. Push to WebSocket
            if prediction:
                await manager.send_to_session(record.session_id, prediction)

    # 5. Also publish to Kafka/Pub/Sub (best-effort, non-blocking)
    try:
        from src.streaming.producer import TOPIC_FCD_RAW, publish

        publish(TOPIC_FCD_RAW, key=record.session_id, value=record.model_dump())
    except Exception:
        pass  # Kafka/Pub/Sub not available — that's fine

    # Periodic cleanup
    sessions.cleanup_expired()

    return IngestResponse(
        status="ok",
        session_id=record.session_id,
        buffer_count=len(session.buffer),
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
