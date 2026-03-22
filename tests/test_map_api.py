"""Tests for map-facing link history endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _road_link() -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        link_id="seoul-link-001",
        road_name="강변북로",
        source="seoul-gis",
        geometry_geojson='{"type":"LineString","coordinates":[[126.9,37.5],[127.0,37.5]]}',
        center_lat=37.5,
        center_lon=126.95,
    )


def _prediction(prediction_id: int = 10) -> SimpleNamespace:
    ts = datetime(2026, 3, 14, 12, 0, tzinfo=UTC)
    return SimpleNamespace(
        id=prediction_id,
        session_id="session-123",
        observed_at=ts,
        density=23.4,
        flow=1450.0,
        fd_density=20.0,
        fd_flow=1300.0,
        residual_density=3.4,
    )


def _fcd_row(t: float) -> SimpleNamespace:
    return SimpleNamespace(time=t, x=t * 10.0, y=0.0, speed=10.0, brake=0.0)


def _create_test_app() -> FastAPI:
    @asynccontextmanager
    async def _test_lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield

    from src.api.map import latest_link_predictions, link_history, prediction_detail

    app = FastAPI(lifespan=_test_lifespan)
    app.add_api_route("/map/links/latest", latest_link_predictions, methods=["GET"])
    app.add_api_route("/map/links/{link_id}/history", link_history, methods=["GET"])
    app.add_api_route("/map/predictions/{prediction_id}", prediction_detail, methods=["GET"])
    return app


def test_latest_link_predictions(monkeypatch) -> None:
    from src.api import map as map_api
    from src.api.database import get_optional_session

    async def _fake_get_session():
        yield object()

    async def _fake_latest(*, session, limit: int = 500):
        assert limit == 500
        return [(_road_link(), _prediction())]

    app = _create_test_app()
    app.dependency_overrides[get_optional_session] = _fake_get_session
    monkeypatch.setattr(map_api, "list_latest_link_predictions", _fake_latest)

    with TestClient(app) as client:
        resp = client.get("/map/links/latest")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["link"]["link_id"] == "seoul-link-001"
    assert data[0]["latest_prediction"]["prediction_id"] == 10


def test_link_history_not_found(monkeypatch) -> None:
    from src.api import map as map_api
    from src.api.database import get_optional_session

    async def _fake_get_session():
        yield object()

    async def _fake_history(*, session, link_id: str, limit: int = 100):
        assert link_id == "missing-link"
        return []

    app = _create_test_app()
    app.dependency_overrides[get_optional_session] = _fake_get_session
    monkeypatch.setattr(map_api, "get_link_history", _fake_history)

    with TestClient(app) as client:
        resp = client.get("/map/links/missing-link/history")

    assert resp.status_code == 404


def test_latest_link_predictions_falls_back_to_demo_on_db_error(monkeypatch) -> None:
    from src.api import map as map_api
    from src.api.database import get_optional_session

    async def _fake_get_session():
        yield object()

    async def _broken_latest(*, session, limit: int = 500):
        del session, limit
        raise RuntimeError("db unavailable")

    demo_link = map_api.RoadLinkSummary(
        link_id="demo-link-001",
        road_name="Demo Road",
        source="demo",
        geometry_geojson='{"type":"LineString","coordinates":[[126.9,37.5],[127.0,37.5]]}',
        center_lat=37.5,
        center_lon=126.95,
    )
    demo_prediction = map_api.LinkPredictionSummary(
        prediction_id=101,
        session_id="demo-session",
        observed_at=datetime(2026, 3, 14, 12, 0, tzinfo=UTC),
        density=12.0,
        flow=800.0,
        fd_density=9.0,
        fd_flow=600.0,
        residual_density=3.0,
    )

    app = _create_test_app()
    app.dependency_overrides[get_optional_session] = _fake_get_session
    monkeypatch.setattr(map_api, "list_latest_link_predictions", _broken_latest)
    monkeypatch.setattr(
        map_api,
        "_demo_payload",
        lambda: ({"demo-link-001": demo_link}, {"demo-link-001": [demo_prediction]}),
    )

    with TestClient(app) as client:
        resp = client.get("/map/links/latest")

    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["link"]["link_id"] == "demo-link-001"
    assert data[0]["latest_prediction"]["prediction_id"] == 101


def test_prediction_detail(monkeypatch) -> None:
    from src.api import map as map_api
    from src.api.database import get_optional_session

    async def _fake_get_session():
        yield object()

    async def _fake_detail(*, session, prediction_id: int):
        assert prediction_id == 10
        return _prediction(prediction_id), _road_link(), [_fcd_row(0.0), _fcd_row(1.0)]

    app = _create_test_app()
    app.dependency_overrides[get_optional_session] = _fake_get_session
    monkeypatch.setattr(map_api, "get_prediction_detail", _fake_detail)

    with TestClient(app) as client:
        resp = client.get("/map/predictions/10")

    assert resp.status_code == 200
    data = resp.json()
    assert data["link"]["road_name"] == "강변북로"
    assert len(data["fcd_records"]) == 2
    assert data["fcd_records"][0]["time"] == 0.0
