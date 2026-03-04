"""Real-time ingestion endpoints: POST /ingest + WebSocket /ws/dashboard."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


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
    kafka_published: bool = False


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

    @property
    def active_sessions(self) -> list[str]:
        return list(self._connections.keys())


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest(record: IngestRecord) -> IngestResponse:
    """Receive a single GPS+accelerometer reading and publish to Kafka.

    The consumer worker will accumulate 300s of data per session,
    run sensor fusion, and invoke the ML prediction pipeline.
    """
    kafka_ok = False
    try:
        from src.streaming.producer import TOPIC_FCD_RAW, publish

        payload = record.model_dump()
        kafka_ok = publish(TOPIC_FCD_RAW, key=record.session_id, value=payload)
    except ImportError:
        logger.debug("Kafka not available — ingest record dropped")
    except Exception:
        logger.warning("Failed to publish ingest record", exc_info=True)

    return IngestResponse(
        status="ok",
        session_id=record.session_id,
        kafka_published=kafka_ok,
    )


# ---------------------------------------------------------------------------
# WebSocket /ws/dashboard/{session_id}
# ---------------------------------------------------------------------------


async def _consume_predictions(session_id: str) -> None:
    """Background task: consume predictions topic and push to WebSocket clients.

    Runs as an asyncio task per WebSocket connection.
    """
    try:
        from kafka import KafkaConsumer

        from src.streaming.producer import TOPIC_PREDICTIONS

        kafka_broker = __import__("os").environ.get("KAFKA_BROKER", "localhost:9092")
        consumer = KafkaConsumer(
            TOPIC_PREDICTIONS,
            bootstrap_servers=kafka_broker,
            group_id=f"ws-{session_id}",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=500,
        )
    except Exception:
        logger.warning("Cannot create Kafka consumer for WebSocket", exc_info=True)
        return

    try:
        while manager.active_sessions and session_id in [
            s for s in manager.active_sessions
        ]:
            for message in consumer:
                data = message.value
                if data.get("session_id") == session_id:
                    await manager.send_to_session(session_id, data)
            await asyncio.sleep(0.5)
    except Exception:
        logger.debug("Prediction consumer stopped for session %s", session_id)
    finally:
        consumer.close()


@router.websocket("/ws/dashboard/{session_id}")
async def dashboard_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint for real-time prediction results.

    Clients connect here to receive density/flow predictions as they
    are produced by the consumer worker.
    """
    await manager.connect(session_id, websocket)

    # Start background consumer for predictions topic
    consumer_task = asyncio.create_task(_consume_predictions(session_id))

    try:
        while True:
            # Keep connection alive; client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
        consumer_task.cancel()
    except Exception:
        manager.disconnect(session_id, websocket)
        consumer_task.cancel()
