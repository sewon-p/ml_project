"""Database CRUD operations for prediction storage."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import Select, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models_db import FCDRecordRow, Prediction, RoadLink, Scenario

logger = logging.getLogger(__name__)


async def save_prediction(
    session: AsyncSession,
    speed_limit: float,
    num_lanes: int,
    fcd_records: list[dict],
    result: dict,
    session_id: str | None = None,
    link_id: str | None = None,
    road_name: str | None = None,
    geometry_geojson: str | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    source: str = "unknown",
    observed_at: datetime | None = None,
) -> int:
    """Save a prediction and its FCD records in a single transaction.

    Returns the prediction ID.
    """
    # 1. Scenario
    scenario = Scenario(speed_limit=speed_limit, num_lanes=num_lanes)
    session.add(scenario)
    await session.flush()  # populate scenario.id

    # 1.5. Road link (optional)
    road_link = None
    if link_id is not None:
        road_link = await upsert_road_link(
            session=session,
            link_id=link_id,
            road_name=road_name,
            geometry_geojson=geometry_geojson,
            center_lat=center_lat,
            center_lon=center_lon,
            source=source,
        )

    # 2. Prediction
    prediction = Prediction(
        scenario_id=scenario.id,
        road_link_id=road_link.id if road_link is not None else None,
        session_id=session_id,
        density=result["density"],
        flow=result["flow"],
        fd_density=result["fd_density"],
        fd_flow=result["fd_flow"],
        residual_density=result["residual_density"],
        observed_at=observed_at or datetime.now(UTC),
    )
    session.add(prediction)
    await session.flush()  # populate prediction.id

    # 3. FCD records (bulk)
    fcd_rows = [
        FCDRecordRow(
            prediction_id=prediction.id,
            time=r["time"],
            x=r["x"],
            y=r["y"],
            speed=r["speed"],
            brake=r.get("brake", 0.0),
        )
        for r in fcd_records
    ]
    session.add_all(fcd_rows)

    await session.commit()
    logger.info(
        "Saved prediction %d (scenario %d, %d fcd rows)",
        prediction.id,
        scenario.id,
        len(fcd_rows),
    )
    return prediction.id


async def upsert_road_link(
    session: AsyncSession,
    link_id: str,
    road_name: str | None = None,
    geometry_geojson: str | None = None,
    center_lat: float | None = None,
    center_lon: float | None = None,
    source: str = "unknown",
) -> RoadLink:
    """Find-or-create road link metadata."""
    stmt = select(RoadLink).where(RoadLink.link_id == link_id)
    road_link = await session.scalar(stmt)
    if road_link is None:
        road_link = RoadLink(
            link_id=link_id,
            road_name=road_name,
            geometry_geojson=geometry_geojson,
            center_lat=center_lat,
            center_lon=center_lon,
            source=source,
        )
        session.add(road_link)
        await session.flush()
        return road_link

    if road_name is not None:
        road_link.road_name = road_name
    if geometry_geojson is not None:
        road_link.geometry_geojson = geometry_geojson
    if center_lat is not None:
        road_link.center_lat = center_lat
    if center_lon is not None:
        road_link.center_lon = center_lon
    if source:
        road_link.source = source
    await session.flush()
    return road_link


def _latest_predictions_stmt(limit: int) -> Select[tuple[RoadLink, Prediction]]:
    latest_per_link = (
        select(
            Prediction.road_link_id.label("road_link_id"),
            func.max(Prediction.observed_at).label("max_observed_at"),
        )
        .where(Prediction.road_link_id.is_not(None))
        .group_by(Prediction.road_link_id)
        .subquery()
    )
    return (
        select(RoadLink, Prediction)
        .join(Prediction, Prediction.road_link_id == RoadLink.id)
        .join(
            latest_per_link,
            (latest_per_link.c.road_link_id == Prediction.road_link_id)
            & (latest_per_link.c.max_observed_at == Prediction.observed_at),
        )
        .order_by(desc(Prediction.observed_at))
        .limit(limit)
    )


async def list_latest_link_predictions(
    session: AsyncSession,
    limit: int = 500,
) -> list[tuple[RoadLink, Prediction]]:
    """Return the latest prediction for each known road link."""
    result = await session.execute(_latest_predictions_stmt(limit))
    return cast(list[tuple[RoadLink, Prediction]], result.all())


async def get_link_history(
    session: AsyncSession,
    link_id: str,
    limit: int = 100,
) -> list[tuple[RoadLink, Prediction]]:
    """Return prediction history for a link, newest first."""
    stmt = (
        select(RoadLink, Prediction)
        .join(Prediction, Prediction.road_link_id == RoadLink.id)
        .where(RoadLink.link_id == link_id)
        .order_by(desc(Prediction.observed_at))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return cast(list[tuple[RoadLink, Prediction]], result.all())


async def get_prediction_detail(
    session: AsyncSession,
    prediction_id: int,
) -> tuple[Prediction, RoadLink | None, list[FCDRecordRow]] | None:
    """Return one prediction with its road link and raw FCD records."""
    prediction = await session.get(Prediction, prediction_id)
    if prediction is None:
        return None

    road_link = None
    if prediction.road_link_id is not None:
        road_link = await session.get(RoadLink, prediction.road_link_id)

    records_stmt = (
        select(FCDRecordRow)
        .where(FCDRecordRow.prediction_id == prediction_id)
        .order_by(FCDRecordRow.time.asc())
    )
    records = list((await session.scalars(records_stmt)).all())
    return prediction, road_link, records
