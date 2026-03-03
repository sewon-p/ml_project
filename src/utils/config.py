"""YAML config loading with deep-merge and override support."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. If it contains a `_base_` key, load and merge with the base."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Support inheritance: _base_ points to a parent config (relative to this file's dir)
    base_path = cfg.pop("_base_", None)
    if base_path is not None:
        base_abs = (path.parent / base_path).resolve()
        base_cfg = load_config(base_abs)
        cfg = merge_configs(base_cfg, cfg)

    return cfg


def merge_configs(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge two dicts. Values in `override` take precedence over `base`."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
