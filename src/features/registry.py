"""Feature registry with singleton pattern and decorator-based registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd


class FeatureRegistry:
    """Stores feature functions in a dict {name: callable}.

    Each registered callable has the signature:
        (trajectory: pd.DataFrame, **kwargs) -> float | dict[str, float]
    """

    def __init__(self) -> None:
        self._features: dict[str, Callable[..., float]] = {}

    def register(self, name: str, func: Callable[..., float]) -> None:
        """Register a feature function under *name*."""
        self._features[name] = func

    def get(self, name: str) -> Callable[..., float]:
        """Return the feature function registered under *name*.

        Raises:
            KeyError: If *name* has not been registered.
        """
        if name not in self._features:
            raise KeyError(f"Feature '{name}' is not registered.")
        return self._features[name]

    def get_subset(self, names: list[str]) -> dict[str, Callable[..., float]]:
        """Return a dict of feature functions for the given *names*."""
        return {name: self.get(name) for name in names}

    def list_all(self) -> list[str]:
        """Return a sorted list of all registered feature names."""
        return sorted(self._features.keys())


# Module-level singleton --------------------------------------------------
_registry = FeatureRegistry()


def register_feature(name: str) -> Callable:
    """Decorator that registers a feature function in the singleton registry.

    If the decorated function returns a ``dict[str, float]``, each entry is
    stored as a separate feature with the key prefixed by *name*.

    Usage::

        @register_feature("speed_mean")
        def speed_mean(trajectory: pd.DataFrame, **kwargs) -> float:
            return float(trajectory["speed"].mean())
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapper(trajectory: pd.DataFrame, **kwargs: Any) -> float | dict[str, float]:
            return func(trajectory, **kwargs)

        _registry.register(name, _wrapper)
        return func

    return decorator
