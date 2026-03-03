"""Tests for config loading and deep merge."""

import pytest
import yaml

from src.utils.config import load_config, merge_configs


class TestMergeConfigs:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"model": {"type": "xgboost", "n_estimators": 100}}
        override = {"model": {"n_estimators": 200}}
        result = merge_configs(base, override)
        assert result == {"model": {"type": "xgboost", "n_estimators": 200}}

    def test_base_not_mutated(self):
        base = {"model": {"lr": 0.01}}
        override = {"model": {"lr": 0.1}}
        merge_configs(base, override)
        assert base["model"]["lr"] == 0.01

    def test_override_replaces_non_dict_with_dict(self):
        base = {"a": 1}
        override = {"a": {"nested": True}}
        result = merge_configs(base, override)
        assert result == {"a": {"nested": True}}


class TestLoadConfig:
    def test_load_simple(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({"seed": 42, "model": {"type": "xgboost"}}))
        cfg = load_config(cfg_file)
        assert cfg["seed"] == 42
        assert cfg["model"]["type"] == "xgboost"

    def test_load_with_base_inheritance(self, tmp_path):
        base_file = tmp_path / "base.yaml"
        base_file.write_text(yaml.dump({"seed": 42, "model": {"type": "xgboost", "lr": 0.01}}))

        child_file = tmp_path / "child.yaml"
        child_file.write_text(yaml.dump({"_base_": "base.yaml", "model": {"lr": 0.1}}))

        cfg = load_config(child_file)
        assert cfg["seed"] == 42
        assert cfg["model"]["type"] == "xgboost"
        assert cfg["model"]["lr"] == 0.1

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_empty_file(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(cfg_file)
        assert cfg == {}
