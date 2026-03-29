"""Map-facing API endpoints for link metadata and prediction history."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import cast

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.crud import get_link_history, get_prediction_detail, list_latest_link_predictions
from src.api.database import get_optional_session
from src.api.models_db import Prediction, RoadLink
from src.api.schemas import (
    FCDRecord,
    LinkHistoryResponse,
    LinkLatestResponse,
    LinkPredictionSummary,
    PredictionDetailResponse,
    RoadLinkSummary,
)
from src.utils.config import load_config

router = APIRouter(prefix="/map", tags=["map"])
logger = logging.getLogger(__name__)


def _serialize_link(road_link: RoadLink) -> RoadLinkSummary:
    return RoadLinkSummary(
        link_id=road_link.link_id,
        road_name=road_link.road_name,
        source=road_link.source,
        geometry_geojson=road_link.geometry_geojson,
        center_lat=road_link.center_lat,
        center_lon=road_link.center_lon,
    )


def _serialize_prediction(prediction: Prediction) -> LinkPredictionSummary:
    return LinkPredictionSummary(
        prediction_id=prediction.id,
        session_id=prediction.session_id,
        observed_at=prediction.observed_at,
        density=prediction.density,
        flow=prediction.flow,
        fd_density=prediction.fd_density,
        fd_flow=prediction.fd_flow,
        residual_density=prediction.residual_density,
    )


@lru_cache(maxsize=1)
def _demo_payload() -> tuple[dict[str, RoadLinkSummary], dict[str, list[LinkPredictionSummary]]]:
    config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
    cfg = load_config(config_path)
    gis_path = cfg.get("gis", {}).get("road_links_path", "data/gis/seoul_links.geojson")
    path = Path(gis_path)
    if not path.is_absolute():
        path = (Path(config_path).resolve().parent.parent / path).resolve()
    if not path.exists():
        return {}, {}

    with open(path, encoding="utf-8") as f:
        geojson = json.load(f)

    links: dict[str, RoadLinkSummary] = {}
    history: dict[str, list[LinkPredictionSummary]] = {}
    base_time = datetime(2026, 3, 14, 16, 0, tzinfo=UTC)

    for idx, feature in enumerate(geojson.get("features", [])[:80], start=1):
        props = feature.get("properties") or {}
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        coords = geometry.get("coordinates", [])
        if geometry.get("type") == "LineString" and coords:
            center_lon = sum(pt[0] for pt in coords) / len(coords)
            center_lat = sum(pt[1] for pt in coords) / len(coords)
        elif geometry.get("type") == "MultiLineString" and coords:
            flat = [pt for line in coords for pt in line]
            center_lon = sum(pt[0] for pt in flat) / len(flat)
            center_lat = sum(pt[1] for pt in flat) / len(flat)
        else:
            continue

        link_id = str(props.get("link_id", f"demo-link-{idx:04d}"))
        link = RoadLinkSummary(
            link_id=link_id,
            road_name=props.get("road_name"),
            source=str(props.get("source", "demo")),
            geometry_geojson=json.dumps(geometry, ensure_ascii=False),
            center_lat=center_lat,
            center_lon=center_lon,
        )
        links[link_id] = link

        entries: list[LinkPredictionSummary] = []
        base_density = 6.0 + (idx % 12) * 4.5
        import math

        for hist_idx in range(8):
            density = base_density + 6.0 * math.sin(hist_idx * 0.8 + idx * 0.3) + hist_idx * 1.2
            density = max(2.0, min(28.0, density))
            fd_density = max(3.0, density - 3.0 - hist_idx * 0.5)
            entries.append(
                LinkPredictionSummary(
                    prediction_id=idx * 100 + hist_idx,
                    session_id=f"demo-session-{idx:03d}",
                    observed_at=base_time - timedelta(minutes=hist_idx * 18 + idx % 11),
                    density=density,
                    flow=density * (8.0 + (idx % 5) * 2.1) * 3.6,
                    fd_density=fd_density,
                    fd_flow=fd_density * (8.0 + (idx % 5) * 2.1) * 3.6,
                    residual_density=density - fd_density,
                )
            )
        history[link_id] = entries

    return links, history


def _demo_fcd(prediction: LinkPredictionSummary) -> list[FCDRecord]:
    speed = max(3.0, prediction.flow / max(prediction.density, 1.0) / 3.6)
    return [
        FCDRecord(
            time=float(t),
            x=float(t) * speed,
            y=0.0,
            speed=speed,
            brake=1.0 if t % 67 == 0 and t > 0 else 0.0,
        )
        for t in range(300)
    ]


def _get_ensemble_snapshot() -> dict:
    """Get ensemble aggregator state for all links."""
    try:
        from src.api.ingest import _get_ensemble_aggregator

        agg = _get_ensemble_aggregator()
        if agg is None:
            return {}
        return {
            lid: {
                "ensemble_density": s.ensemble_density,
                "ensemble_flow": s.ensemble_flow,
                "probe_count": s.probe_count,
                "is_frozen": s.is_frozen,
            }
            for lid, s in agg._active.items()
        }
    except Exception:
        return {}


def _live_payload() -> tuple[dict[str, RoadLinkSummary], dict[str, list[LinkPredictionSummary]]]:
    try:
        from src.api.ingest import get_live_map_snapshot

        snapshot = get_live_map_snapshot()
    except Exception:
        snapshot = {}

    links: dict[str, RoadLinkSummary] = {}
    history: dict[str, list[LinkPredictionSummary]] = {}
    for link_id, payload in snapshot.items():
        link = payload["link"]
        links[link_id] = RoadLinkSummary(**link)
        history[link_id] = [
            LinkPredictionSummary(
                prediction_id=item["prediction_id"],
                session_id=item["session_id"],
                observed_at=item["observed_at"],
                density=item["density"],
                flow=item["flow"],
                fd_density=item["fd_density"],
                fd_flow=item["fd_flow"],
                residual_density=item["residual_density"],
            )
            for item in payload["history"]
        ]
    return links, history


@router.get("/links/latest", response_model=list[LinkLatestResponse])
async def latest_link_predictions(
    limit: int = 500,
    session: AsyncSession | None = Depends(get_optional_session),
) -> list[LinkLatestResponse]:
    live_links, live_history = _live_payload()
    if live_links:
        ensemble_snapshot = _get_ensemble_snapshot()
        results = []
        for link_id in list(live_links.keys())[:limit]:
            if link_id not in live_history or not live_history[link_id]:
                continue
            ens = ensemble_snapshot.get(link_id)
            ens_summary = None
            if ens and ens.get("probe_count", 0) > 0:
                from src.api.schemas import EnsembleSummary

                ens_summary = EnsembleSummary(
                    ensemble_density=ens["ensemble_density"],
                    ensemble_flow=ens.get("ensemble_flow", 0.0),
                    probe_count=ens["probe_count"],
                    is_frozen=ens.get("is_frozen", False),
                )
            results.append(
                LinkLatestResponse(
                    link=live_links[link_id],
                    latest_prediction=live_history[link_id][0],
                    ensemble=ens_summary,
                )
            )
        return results

    if session is not None:
        try:
            rows = await list_latest_link_predictions(session=session, limit=limit)
            if rows:
                return [
                    LinkLatestResponse(
                        link=_serialize_link(road_link),
                        latest_prediction=_serialize_prediction(prediction),
                    )
                    for road_link, prediction in rows
                ]
        except Exception:
            logger.warning("Latest link query failed; falling back to demo payload", exc_info=True)

    links, history = _demo_payload()
    return [
        LinkLatestResponse(link=link, latest_prediction=history[link_id][0])
        for link_id, link in list(links.items())[:limit]
        if link_id in history and history[link_id]
    ]


@router.get("/links/{link_id}/history", response_model=LinkHistoryResponse)
async def link_history(
    link_id: str,
    limit: int = 100,
    session: AsyncSession | None = Depends(get_optional_session),
) -> LinkHistoryResponse:
    live_links, live_history = _live_payload()
    if link_id in live_links and link_id in live_history:
        ens_snap = _get_ensemble_snapshot().get(link_id)
        ens_summary = None
        if ens_snap and ens_snap.get("probe_count", 0) > 0:
            from src.api.schemas import EnsembleSummary

            ens_summary = EnsembleSummary(
                ensemble_density=ens_snap["ensemble_density"],
                ensemble_flow=ens_snap.get("ensemble_flow", 0.0),
                probe_count=ens_snap["probe_count"],
            )
        return LinkHistoryResponse(
            link=live_links[link_id],
            history=live_history[link_id][:limit],
            ensemble=ens_summary,
        )

    if session is not None:
        try:
            rows = await get_link_history(session=session, link_id=link_id, limit=limit)
            if rows:
                road_link = rows[0][0]
                return LinkHistoryResponse(
                    link=_serialize_link(road_link),
                    history=[_serialize_prediction(prediction) for _, prediction in rows],
                )
        except Exception:
            logger.warning("Link history query failed; falling back to demo payload", exc_info=True)

    links, history = _demo_payload()
    if link_id not in links or link_id not in history:
        raise HTTPException(status_code=404, detail=f"Unknown link_id: {link_id}")
    return LinkHistoryResponse(link=links[link_id], history=history[link_id][:limit])


@router.get("/predictions/{prediction_id}", response_model=PredictionDetailResponse)
async def prediction_detail(
    prediction_id: int,
    session: AsyncSession | None = Depends(get_optional_session),
) -> PredictionDetailResponse:
    try:
        from src.api.ingest import get_live_prediction_detail

        live_detail = get_live_prediction_detail(prediction_id)
    except Exception:
        live_detail = None
    if live_detail is not None:
        link_meta, entry = live_detail
        link = RoadLinkSummary(**link_meta)
        summary = LinkPredictionSummary(
            prediction_id=entry["prediction_id"],
            session_id=entry["session_id"],
            observed_at=entry["observed_at"],
            density=entry["density"],
            flow=entry["flow"],
            fd_density=entry["fd_density"],
            fd_flow=entry["fd_flow"],
            residual_density=entry["residual_density"],
        )
        return PredictionDetailResponse(
            prediction_id=summary.prediction_id,
            link=link,
            session_id=summary.session_id,
            observed_at=summary.observed_at,
            density=summary.density,
            flow=summary.flow,
            fd_density=summary.fd_density,
            fd_flow=summary.fd_flow,
            residual_density=summary.residual_density,
            fcd_records=_demo_fcd(summary),
        )

    if session is not None:
        try:
            detail = await get_prediction_detail(session=session, prediction_id=prediction_id)
            if detail is not None:
                prediction, road_link, records = detail
                return PredictionDetailResponse(
                    prediction_id=prediction.id,
                    link=_serialize_link(road_link) if road_link is not None else None,
                    session_id=prediction.session_id,
                    observed_at=prediction.observed_at,
                    density=prediction.density,
                    flow=prediction.flow,
                    fd_density=prediction.fd_density,
                    fd_flow=prediction.fd_flow,
                    residual_density=prediction.residual_density,
                    fcd_records=[
                        FCDRecord(
                            time=record.time,
                            x=record.x,
                            y=record.y,
                            speed=record.speed,
                            brake=record.brake,
                        )
                        for record in records
                    ],
                )
        except Exception:
            logger.warning(
                "Prediction detail query failed; falling back to demo payload", exc_info=True
            )

    links, history = _demo_payload()
    for link_id, entries in history.items():
        for demo_prediction in entries:
            demo_entry = cast(LinkPredictionSummary, demo_prediction)
            if demo_entry.prediction_id == prediction_id:
                return PredictionDetailResponse(
                    prediction_id=demo_entry.prediction_id,
                    link=links.get(link_id),
                    session_id=demo_entry.session_id,
                    observed_at=demo_entry.observed_at,
                    density=demo_entry.density,
                    flow=demo_entry.flow,
                    fd_density=demo_entry.fd_density,
                    fd_flow=demo_entry.fd_flow,
                    residual_density=demo_entry.residual_density,
                    fcd_records=_demo_fcd(demo_entry),
                )
    raise HTTPException(status_code=404, detail=f"Unknown prediction_id: {prediction_id}")
