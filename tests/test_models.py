"""Tests for model definitions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.base import BaseEstimator
from src.models.cnn1d import CNN1D, CNN1DEstimator
from src.models.factory import create_model
from src.models.fd_baseline import FDBaselineEstimator
from src.models.lstm import LSTMEstimator, LSTMModel


class TestCNN1D:
    def test_output_shape(self):
        model = CNN1D(in_channels=6, seq_len=60)
        x = torch.randn(4, 6, 60)
        out = model(x)
        assert out.shape == (4,)

    def test_estimator_predict(self):
        est = CNN1DEstimator(in_channels=6, seq_len=60)
        X = np.random.randn(4, 6, 60).astype(np.float32)
        preds = est.predict(X)
        assert preds.shape == (4,)


class TestLSTM:
    def test_output_shape(self):
        model = LSTMModel(input_size=6, hidden_size=32, num_layers=1)
        # Input: (B, C, L) — same canonical format as CNN1D
        x = torch.randn(4, 6, 60)
        out = model(x)
        assert out.shape == (4,)

    def test_estimator_predict(self):
        est = LSTMEstimator(input_size=6, hidden_size=32, num_layers=1)
        # (N, C, L) format
        X = np.random.randn(4, 6, 60).astype(np.float32)
        preds = est.predict(X)
        assert preds.shape == (4,)


class TestFDBaseline:
    def test_predict_density(self):
        est = FDBaselineEstimator(v_free=30.0, k_jam=120.0)
        speeds = np.array([[30.0], [15.0], [0.0]])
        density = est.predict(speeds)
        assert density[0] == pytest.approx(0.0, abs=0.1)
        assert density[1] == pytest.approx(60.0, abs=0.1)
        assert density[2] == pytest.approx(120.0, abs=0.1)

    def test_predict_flow(self):
        est = FDBaselineEstimator(v_free=30.0, k_jam=120.0)
        speeds = np.array([[0.0], [30.0]])
        flow = est.predict_flow(speeds)
        assert flow[0] == pytest.approx(0.0, abs=0.1)
        assert flow[1] == pytest.approx(0.0, abs=0.1)

    def test_save_load(self, tmp_path):
        est = FDBaselineEstimator(v_free=25.0, k_jam=100.0)
        path = est.save(tmp_path / "fd.pkl")
        loaded = FDBaselineEstimator.load(path)
        assert loaded.v_free == 25.0
        assert loaded.k_jam == 100.0


class TestFactory:
    def test_create_fd_baseline(self):
        model = create_model("fd_baseline", v_free=30.0, k_jam=120.0)
        assert isinstance(model, BaseEstimator)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("nonexistent_model")
