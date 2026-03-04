"""Kafka consumer worker: accumulates 300s FCD windows and runs prediction."""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
_WINDOW_SIZE = 300  # 300 seconds of FCD data
_SESSION_TIMEOUT = 600  # 10 minutes inactivity → delete session
_CONSUMER_GROUP = "fcd-consumer-group"


@dataclass
class SessionState:
    """Per-session sliding window of FCD records."""

    session_id: str
    buffer: deque[dict] = field(default_factory=lambda: deque(maxlen=_WINDOW_SIZE))
    last_active: float = field(default_factory=time.time)
    prediction_count: int = 0

    def add(self, record: dict) -> None:
        """Add a fused FCD record to the buffer."""
        self.buffer.append(record)
        self.last_active = time.time()

    @property
    def ready(self) -> bool:
        """True when buffer has enough data for prediction."""
        return len(self.buffer) >= _WINDOW_SIZE

    @property
    def is_expired(self) -> bool:
        """True if session has been inactive beyond timeout."""
        return (time.time() - self.last_active) > _SESSION_TIMEOUT

    def get_window(self) -> list[dict]:
        """Return the current 300-record window as a list."""
        return list(self.buffer)


class StreamConsumer:
    """Kafka consumer that processes FCD streams and produces predictions.

    Usage:
        consumer = StreamConsumer()
        consumer.run()  # blocks until SIGTERM/SIGINT
    """

    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}
        self._running = False
        self._registry = None
        self._fusion_cache: dict[str, object] = {}

    def _load_model(self) -> None:
        """Load the ML model registry (reuses existing code)."""
        from src.api.dependencies import ModelRegistry

        config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
        self._registry = ModelRegistry(config_path)
        logger.info(
            "Model loaded: %s (%d features)",
            self._registry.model_path,
            len(self._registry.feature_columns),
        )

    def _get_or_create_session(self, session_id: str) -> SessionState:
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
            logger.info("New session: %s", session_id)
        return self.sessions[session_id]

    def _cleanup_expired_sessions(self) -> None:
        """Remove sessions that have been inactive beyond timeout."""
        expired = [sid for sid, s in self.sessions.items() if s.is_expired]
        for sid in expired:
            del self.sessions[sid]
            self._fusion_cache.pop(sid, None)
            logger.info("Expired session: %s", sid)

    def _predict(self, session: SessionState, speed_limit: float, num_lanes: int) -> dict | None:
        """Run prediction on a session's FCD window."""
        if self._registry is None:
            return None

        from src.api.inference import predict_density

        try:
            fcd_records = session.get_window()
            result = predict_density(
                fcd_records=fcd_records,
                speed_limit=speed_limit,
                num_lanes=num_lanes,
                registry=self._registry,
            )
            session.prediction_count += 1
            return result
        except Exception:
            logger.warning("Prediction failed for session %s", session.session_id, exc_info=True)
            return None

    def _publish_prediction(
        self, session_id: str, result: dict, timestamp: float
    ) -> None:
        """Publish prediction result to the predictions topic."""
        from src.streaming.producer import TOPIC_PREDICTIONS, publish

        message = {
            "session_id": session_id,
            "timestamp": timestamp,
            **result,
        }
        publish(TOPIC_PREDICTIONS, key=session_id, value=message)

    def _process_message(self, message: dict) -> None:
        """Process a single consumed FCD message."""
        from src.streaming.fusion import SensorFusion

        session_id = message.get("session_id", "unknown")
        session = self._get_or_create_session(session_id)

        # Get or create fusion state for this session
        if session_id not in self._fusion_cache:
            self._fusion_cache[session_id] = SensorFusion()
        fusion: SensorFusion = self._fusion_cache[session_id]  # type: ignore[assignment]

        # Fuse raw sensor data → FCD record
        fcd = fusion.process(
            lat=message.get("lat", 0.0),
            lon=message.get("lon", 0.0),
            speed=message.get("speed", 0.0),
            heading=message.get("heading", 0.0),
            accel_x=message.get("accel_x", 0.0),
            accel_y=message.get("accel_y", 0.0),
            accel_z=message.get("accel_z", 0.0),
            timestamp=message.get("timestamp", time.time()),
        )
        session.add(fcd)

        # Predict when buffer is full
        if session.ready:
            speed_limit = message.get("speed_limit", 22.22)  # default 80 km/h
            num_lanes = message.get("num_lanes", 2)
            result = self._predict(session, speed_limit, num_lanes)
            if result:
                self._publish_prediction(session_id, result, fcd["time"])
                logger.debug(
                    "Prediction #%d for %s: density=%.2f flow=%.1f",
                    session.prediction_count,
                    session_id,
                    result["density"],
                    result["flow"],
                )

    def run(self) -> None:
        """Main consumer loop. Blocks until SIGTERM/SIGINT."""
        from kafka import KafkaConsumer

        from src.streaming.producer import TOPIC_FCD_RAW

        # Load ML model
        self._load_model()

        # Setup signal handlers for graceful shutdown
        self._running = True

        def _shutdown(signum: int, frame: object) -> None:
            logger.info("Received signal %d, shutting down...", signum)
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        # Create Kafka consumer
        consumer = KafkaConsumer(
            TOPIC_FCD_RAW,
            bootstrap_servers=_KAFKA_BROKER,
            group_id=_CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
        )
        logger.info("Consumer started, listening on topic: %s", TOPIC_FCD_RAW)

        cleanup_interval = 60  # seconds
        last_cleanup = time.time()

        try:
            while self._running:
                # Poll messages (returns within consumer_timeout_ms)
                for message in consumer:
                    if not self._running:
                        break
                    try:
                        self._process_message(message.value)
                    except Exception:
                        logger.warning("Error processing message", exc_info=True)

                # Periodic session cleanup
                now = time.time()
                if now - last_cleanup > cleanup_interval:
                    self._cleanup_expired_sessions()
                    last_cleanup = now
        finally:
            consumer.close()
            logger.info("Consumer stopped")


def main() -> None:
    """Entry point for running the consumer as a standalone process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    consumer = StreamConsumer()
    consumer.run()


if __name__ == "__main__":
    main()
