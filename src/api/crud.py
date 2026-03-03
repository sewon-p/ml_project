"""Database CRUD operations for prediction storage."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models_db import FCDRecordRow, Prediction, Scenario

logger = logging.getLogger(__name__)


async def save_prediction(
    session: AsyncSession,
    speed_limit: float,
    num_lanes: int,
    fcd_records: list[dict],
    result: dict,
) -> int:
    """Save a prediction and its FCD records in a single transaction.

    Returns the prediction ID.
    """
    # 1. Scenario
    scenario = Scenario(speed_limit=speed_limit, num_lanes=num_lanes)
    session.add(scenario)
    await session.flush()  # populate scenario.id

    # 2. Prediction
    prediction = Prediction(
        scenario_id=scenario.id,
        density=result["density"],
        flow=result["flow"],
        fd_density=result["fd_density"],
        fd_flow=result["fd_flow"],
        residual_density=result["residual_density"],
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
