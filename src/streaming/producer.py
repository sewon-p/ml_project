"""Message broker abstraction: Kafka (local/Docker) or GCP Pub/Sub (Cloud Run).

Auto-detection:
  - PUBSUB_PROJECT env → GCP Pub/Sub
  - KAFKA_BROKER env  → Kafka
  - Neither           → disabled (publish() returns False)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Topic names (used as Kafka topic or Pub/Sub topic)
TOPIC_FCD_RAW = "fcd-raw"
TOPIC_PREDICTIONS = "predictions"


# ---------------------------------------------------------------------------
# Abstract broker
# ---------------------------------------------------------------------------


class MessageBroker(ABC):
    """Abstract message broker interface."""

    @abstractmethod
    def publish(self, topic: str, key: str, value: dict) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Kafka broker
# ---------------------------------------------------------------------------


class KafkaBroker(MessageBroker):
    """Kafka producer using kafka-python."""

    def __init__(self, broker: str) -> None:
        from kafka import KafkaProducer

        self._producer = KafkaProducer(
            bootstrap_servers=broker,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            max_block_ms=5000,
        )
        logger.info("Kafka producer connected to %s", broker)

    def publish(self, topic: str, key: str, value: dict) -> bool:
        try:
            self._producer.send(topic, key=key, value=value)
            self._producer.flush(timeout=2.0)
            return True
        except Exception:
            logger.warning("Failed to publish to Kafka topic %s", topic, exc_info=True)
            return False

    def close(self) -> None:
        try:
            self._producer.close(timeout=5)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GCP Pub/Sub broker
# ---------------------------------------------------------------------------


class PubSubBroker(MessageBroker):
    """GCP Pub/Sub publisher using google-cloud-pubsub."""

    def __init__(self, project_id: str) -> None:
        from google.cloud import pubsub_v1

        self._publisher = pubsub_v1.PublisherClient()
        self._project_id = project_id
        self._topic_cache: dict[str, str] = {}
        logger.info("Pub/Sub publisher initialized for project: %s", project_id)

    def _topic_path(self, topic: str) -> str:
        if topic not in self._topic_cache:
            self._topic_cache[topic] = self._publisher.topic_path(self._project_id, topic)
            # Ensure topic exists
            try:
                self._publisher.get_topic(topic=self._topic_cache[topic])
            except Exception:
                self._publisher.create_topic(name=self._topic_cache[topic])
                logger.info("Created Pub/Sub topic: %s", topic)
        return self._topic_cache[topic]

    def publish(self, topic: str, key: str, value: dict) -> bool:
        try:
            topic_path = self._topic_path(topic)
            data = json.dumps(value).encode("utf-8")
            future = self._publisher.publish(
                topic_path,
                data,
                session_id=key,  # attribute for filtering
            )
            future.result(timeout=5.0)
            return True
        except Exception:
            logger.warning("Failed to publish to Pub/Sub topic %s", topic, exc_info=True)
            return False

    def close(self) -> None:
        try:
            self._publisher.transport.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_broker: MessageBroker | None = None
_init_attempted = False


def get_broker() -> MessageBroker | None:
    """Get or create the message broker singleton.

    Auto-detects backend:
      - PUBSUB_PROJECT env var → GCP Pub/Sub
      - KAFKA_BROKER env var   → Kafka
      - Neither                → None (disabled)
    """
    global _broker, _init_attempted
    if _broker is not None or _init_attempted:
        return _broker

    _init_attempted = True

    pubsub_project = os.environ.get("PUBSUB_PROJECT")
    kafka_broker = os.environ.get("KAFKA_BROKER")

    if pubsub_project:
        try:
            _broker = PubSubBroker(pubsub_project)
        except Exception:
            logger.warning("Pub/Sub initialization failed", exc_info=True)
    elif kafka_broker:
        try:
            _broker = KafkaBroker(kafka_broker)
        except Exception:
            logger.warning("Kafka initialization failed", exc_info=True)
    else:
        logger.info("No message broker configured (set PUBSUB_PROJECT or KAFKA_BROKER)")

    return _broker


def publish(topic: str, key: str, value: dict) -> bool:
    """Publish a message to the configured broker.

    Args:
        topic: Topic name (e.g., 'fcd-raw', 'predictions').
        key: Message key (session_id) for ordering/partitioning.
        value: Message payload (JSON-serializable dict).

    Returns:
        True if published, False if broker unavailable or publish failed.
    """
    broker = get_broker()
    if broker is None:
        return False
    return broker.publish(topic, key, value)


def close() -> None:
    """Close the broker connection (call on shutdown)."""
    global _broker, _init_attempted
    if _broker is not None:
        _broker.close()
        _broker = None
    _init_attempted = False
