"""Tests for the FastAPI inference API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_fcd_records(n: int = 300, speed_base: float = 20.0, seed: int = 42) -> list[dict]:
    """Generate synthetic FCD records for testing."""
    rng = np.random.RandomState(seed)
    records = []
    x = 0.0
    for i in range(n):
        speed = max(0.1, speed_base + rng.randn() * 1.0)
        x += speed * 1.0  # dt=1s
        records.append(
            {
                "time": float(i),
                "x": x,
                "y": rng.randn() * 0.1,
                "speed": speed,
                "brake": 0.0,
            }
        )
    return records


def _make_mock_registry() -> MagicMock:
    """Create a mock ModelRegistry."""
    registry = MagicMock()
    registry.model_path = "outputs/xgboost_best.pkl"
    registry.residual_enabled = True
    registry.v_free_factor = 1.1
    registry.vehicle_length = 4.5
    registry.min_gap = 2.5
    registry.features_drop = {
        "vx_mean",
        "vx_std",
        "vx_min",
        "vx_max",
        "vx_autocorr_lag1",
        "vx_fft_dominant_freq",
        "harsh_accel_count",
        "harsh_decel_count",
        "lane_change_count",
    }
    registry.feature_columns = [
        "speed_mean",
        "speed_std",
        "speed_cv",
        "speed_iqr",
        "speed_min",
        "speed_max",
        "speed_median",
        "speed_p10",
        "speed_p90",
        "num_lanes",
        "speed_limit",
    ]
    # predict returns a small residual
    registry.model.predict.return_value = np.array([2.5])
    return registry


def _create_test_app(registry: MagicMock) -> FastAPI:
    """Build a FastAPI app with a mock registry (no real model loading, no DB)."""

    @asynccontextmanager
    async def _test_lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.registry = registry
        app.state.db_available = False
        yield

    from src.api.app import health, predict

    test_app = FastAPI(lifespan=_test_lifespan)
    test_app.add_api_route("/predict", predict, methods=["POST"])
    test_app.add_api_route("/health", health, methods=["GET"])
    return test_app


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with a mocked ModelRegistry."""
    mock_registry = _make_mock_registry()
    test_app = _create_test_app(mock_registry)
    with TestClient(test_app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "xgboost"
        assert data["model_path"] == "outputs/xgboost_best.pkl"
        assert data["residual_correction"] is True
        assert data["n_features"] == 11


class TestPredictEndpoint:
    def test_predict_returns_density_and_flow(self, client: TestClient) -> None:
        payload = {
            "fcd_records": _make_fcd_records(300),
            "speed_limit": 22.22,
            "num_lanes": 2,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "density" in data
        assert "flow" in data
        assert "fd_density" in data
        assert "fd_flow" in data
        assert "residual_density" in data
        assert "prediction_id" in data
        # prediction_id is None when DB is unavailable
        assert data["prediction_id"] is None
        # residual should match mock
        assert data["residual_density"] == pytest.approx(2.5)
        # density = fd_density + residual
        assert data["density"] == pytest.approx(data["fd_density"] + 2.5)

    def test_predict_empty_records(self, client: TestClient) -> None:
        payload = {
            "fcd_records": [],
            "speed_limit": 22.22,
            "num_lanes": 2,
        }
        resp = client.post("/predict", json=payload)
        # build_trajectory returns empty DF for < 2 rows → feature extraction will fail
        assert resp.status_code == 500

    def test_predict_missing_field(self, client: TestClient) -> None:
        payload = {
            "fcd_records": _make_fcd_records(10),
            # missing speed_limit and num_lanes
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422  # validation error


class TestSchemas:
    def test_fcd_record_defaults(self) -> None:
        from src.api.schemas import FCDRecord

        record = FCDRecord(time=0.0, x=1.0, y=2.0, speed=10.0)
        assert record.brake == 0.0

    def test_predict_request_validation(self) -> None:
        from src.api.schemas import PredictRequest

        req = PredictRequest(
            fcd_records=[
                {"time": 0, "x": 0, "y": 0, "speed": 10},
            ],
            speed_limit=22.22,
            num_lanes=2,
        )
        assert len(req.fcd_records) == 1
        assert req.num_lanes == 2


class TestInference:
    def test_predict_density_returns_expected_keys(self) -> None:
        from src.api.inference import predict_density

        registry = _make_mock_registry()
        fcd = _make_fcd_records(300)
        result = predict_density(
            fcd_records=fcd,
            speed_limit=22.22,
            num_lanes=2,
            registry=registry,
        )
        # predict_density returns core result keys (prediction_id is added by the endpoint)
        expected_keys = {"density", "flow", "fd_density", "fd_flow", "residual_density"}
        assert set(result.keys()) == expected_keys
        # All values should be finite
        for v in result.values():
            assert np.isfinite(v)

    def test_predict_density_residual_adds_to_fd(self) -> None:
        from src.api.inference import predict_density

        registry = _make_mock_registry()
        registry.model.predict.return_value = np.array([5.0])
        fcd = _make_fcd_records(300)
        result = predict_density(
            fcd_records=fcd,
            speed_limit=22.22,
            num_lanes=2,
            registry=registry,
        )
        assert result["density"] == pytest.approx(result["fd_density"] + 5.0)


class TestBuildTrajectory:
    def test_build_trajectory_output_shape(self) -> None:
        import pandas as pd

        from src.data.preprocessing import build_trajectory

        n = 300
        rng = np.random.RandomState(0)
        raw = pd.DataFrame(
            {
                "time": np.arange(n, dtype=float),
                "x": np.cumsum(rng.uniform(10, 30, n)),
                "y": rng.randn(n) * 0.1,
                "speed": rng.uniform(10, 30, n),
                "brake": np.zeros(n),
            }
        )
        traj = build_trajectory(raw)
        assert len(traj) == n
        assert list(traj.columns) == ["VX", "VY", "AX", "AY", "speed", "brake"]

    def test_build_trajectory_infers_brake(self) -> None:
        import pandas as pd

        from src.data.preprocessing import build_trajectory

        raw = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "x": [0.0, 20.0, 40.0, 45.0, 44.0],  # decelerating at end
                "y": [0.0, 0.0, 0.0, 0.0, 0.0],
                "speed": [20.0, 20.0, 20.0, 5.0, -1.0],
            }
        )
        traj = build_trajectory(raw)
        assert "brake" in traj.columns
        assert len(traj) == 5
