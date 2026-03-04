"""Consumer worker: accumulates 300s FCD windows and runs prediction.

Supports both Kafka (local/Docker) and GCP Pub/Sub (Cloud Run) backends.
Auto-detects via PUBSUB_PROJECT / KAFKA_BROKER environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

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
    """Consumer that processes FCD streams and produces predictions.

    Auto-detects Kafka or Pub/Sub based on environment variables.

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

    def _publish_prediction(self, session_id: str, result: dict, timestamp: float) -> None:
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

        # Fuse raw sensor data → FCD record (with Kalman filter)
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

    # ------------------------------------------------------------------
    # Kafka consumer loop
    # ------------------------------------------------------------------

    def _run_kafka(self) -> None:
        """Consume from Kafka."""
        from kafka import KafkaConsumer

        from src.streaming.producer import TOPIC_FCD_RAW

        kafka_broker = os.environ.get("KAFKA_BROKER", "localhost:9092")
        consumer = KafkaConsumer(
            TOPIC_FCD_RAW,
            bootstrap_servers=kafka_broker,
            group_id=_CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            consumer_timeout_ms=1000,
        )
        logger.info("Kafka consumer started, topic: %s", TOPIC_FCD_RAW)

        cleanup_interval = 60
        last_cleanup = time.time()

        try:
            while self._running:
                for message in consumer:
                    if not self._running:
                        break
                    try:
                        self._process_message(message.value)
                    except Exception:
                        logger.warning("Error processing message", exc_info=True)

                now = time.time()
                if now - last_cleanup > cleanup_interval:
                    self._cleanup_expired_sessions()
                    last_cleanup = now
        finally:
            consumer.close()

    # ------------------------------------------------------------------
    # GCP Pub/Sub consumer loop
    # ------------------------------------------------------------------

    def _run_pubsub(self) -> None:
        """Consume from GCP Pub/Sub using streaming pull."""
        from google.cloud import pubsub_v1

        from src.streaming.producer import TOPIC_FCD_RAW

        project_id = os.environ["PUBSUB_PROJECT"]
        subscriber = pubsub_v1.SubscriberClient()
        subscription_id = f"{TOPIC_FCD_RAW}-sub"
        subscription_path = subscriber.subscription_path(project_id, subscription_id)

        # Ensure subscription exists
        topic_path = subscriber.topic_path(project_id, TOPIC_FCD_RAW)  # type: ignore[attr-defined]
        try:
            subscriber.get_subscription(subscription=subscription_path)
        except Exception:
            subscriber.create_subscription(
                name=subscription_path,
                topic=topic_path,
                ack_deadline_seconds=30,
            )
            logger.info("Created Pub/Sub subscription: %s", subscription_id)

        logger.info("Pub/Sub consumer started, subscription: %s", subscription_id)

        cleanup_interval = 60
        last_cleanup = time.time()

        def callback(message: object) -> None:
            try:
                data = json.loads(message.data.decode("utf-8"))  # type: ignore[union-attr]
                self._process_message(data)
                message.ack()  # type: ignore[union-attr]
            except Exception:
                logger.warning("Error processing Pub/Sub message", exc_info=True)
                message.nack()  # type: ignore[union-attr]

        streaming_pull = subscriber.subscribe(subscription_path, callback=callback)

        try:
            while self._running:
                time.sleep(1)
                now = time.time()
                if now - last_cleanup > cleanup_interval:
                    self._cleanup_expired_sessions()
                    last_cleanup = now
        finally:
            streaming_pull.cancel()
            streaming_pull.result(timeout=5)
            subscriber.close()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main consumer loop. Auto-detects Kafka or Pub/Sub. Blocks until SIGTERM/SIGINT."""
        self._load_model()
        self._running = True

        def _shutdown(signum: int, frame: object) -> None:
            logger.info("Received signal %d, shutting down...", signum)
            self._running = False

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        pubsub_project = os.environ.get("PUBSUB_PROJECT")
        kafka_broker = os.environ.get("KAFKA_BROKER")

        try:
            if pubsub_project:
                logger.info("Using GCP Pub/Sub backend (project: %s)", pubsub_project)
                self._run_pubsub()
            elif kafka_broker:
                logger.info("Using Kafka backend (broker: %s)", kafka_broker)
                self._run_kafka()
            else:
                logger.error("No broker configured. Set PUBSUB_PROJECT or KAFKA_BROKER.")
        finally:
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
