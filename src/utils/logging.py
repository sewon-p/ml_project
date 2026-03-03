"""Structured logging setup with console and optional file output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "ml-project",
    level: int | str = logging.INFO,
    log_dir: str | Path | None = None,
    filename: str = "train.log",
) -> logging.Logger:
    """Create a logger with console handler and optional file handler.

    Args:
        name: Logger name.
        level: Logging level (int or string like "DEBUG").
        log_dir: If provided, also write logs to ``log_dir/filename``.
        filename: Log file name inside *log_dir*.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (stderr so stdout stays clean for piping)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / filename, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
