"""Tests for the feature registry."""

from __future__ import annotations

import pytest

from src.features.registry import FeatureRegistry


def _dummy_mean(trajectory, **kw):
    return float(trajectory["speed"].mean())


def _const_1(trajectory, **kw):
    return 1.0


def _const_2(trajectory, **kw):
    return 2.0


def _const_3(trajectory, **kw):
    return 3.0


def _const_0(trajectory, **kw):
    return 0.0


class TestFeatureRegistry:
    def test_register_and_get(self):
        reg = FeatureRegistry()
        reg.register("dummy_mean", _dummy_mean)
        assert reg.get("dummy_mean") is _dummy_mean

    def test_get_missing_raises(self):
        reg = FeatureRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_get_subset(self):
        reg = FeatureRegistry()
        reg.register("a", _const_1)
        reg.register("b", _const_2)
        reg.register("c", _const_3)
        subset = reg.get_subset(["a", "c"])
        assert set(subset.keys()) == {"a", "c"}

    def test_list_all(self):
        reg = FeatureRegistry()
        reg.register("z_feat", _const_0)
        reg.register("a_feat", _const_0)
        names = reg.list_all()
        assert names == ["a_feat", "z_feat"]
