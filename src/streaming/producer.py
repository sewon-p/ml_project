"""Kafka producer singleton for publishing FCD records."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_producer: Any = None
_KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")

# Topic names
TOPIC_FCD_RAW = "fcd-raw"
TOPIC_PREDICTIONS = "predictions"


def get_producer() -> Any:
    """Get or create the Kafka producer singleton.

    Returns None if kafka-python is not installed or broker is unavailable.
    """
    global _producer
    if _producer is not None:
        return _producer

    try:
        from kafka import KafkaProducer

        _producer = KafkaProducer(
            bootstrap_servers=_KAFKA_BROKER,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retries=3,
            max_block_ms=5000,
        )
        logger.info("Kafka producer connected to %s", _KAFKA_BROKER)
    except Exception:
        logger.warning("Kafka producer unavailable — streaming disabled", exc_info=True)
        _producer = None
    return _producer


def publish(topic: str, key: str, value: dict) -> bool:
    """Publish a message to a Kafka topic.

    Args:
        topic: Kafka topic name.
        key: Message key (e.g., session_id for partition affinity).
        value: Message payload (will be JSON-serialized).

    Returns:
        True if published successfully, False otherwise.
    """
    producer = get_producer()
    if producer is None:
        return False
    try:
        producer.send(topic, key=key, value=value)
        producer.flush(timeout=2.0)
        return True
    except Exception:
        logger.warning("Failed to publish to %s", topic, exc_info=True)
        return False


def close() -> None:
    """Close the Kafka producer (call on shutdown)."""
    global _producer
    if _producer is not None:
        try:
            _producer.close(timeout=5)
        except Exception:
            pass
        _producer = None
