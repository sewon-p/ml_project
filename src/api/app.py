"""FastAPI application for traffic density estimation."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import ModelRegistry
from src.api.inference import predict_density
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


def _get_database_url() -> str | None:
    """Resolve DATABASE_URL from environment or config."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
    try:
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("database", {}).get("url")
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the XGBoost model and initialize database at startup."""
    config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
    app.state.registry = ModelRegistry(config_path)
    logger.info(
        "Model loaded: %s (%d features)",
        app.state.registry.model_path,
        len(app.state.registry.feature_columns),
    )

    # Initialize database (optional — prediction works without DB)
    database_url = _get_database_url()
    app.state.db_available = False
    if database_url:
        try:
            from src.api.database import init_db

            await init_db(database_url)
            app.state.db_available = True
            logger.info("Database connected")
        except Exception:
            logger.warning("Database initialization failed — running without DB", exc_info=True)

    yield

    # Shutdown: close database connection pool
    if app.state.db_available:
        try:
            from src.api.database import close_db

            await close_db()
        except Exception:
            logger.warning("Error closing database", exc_info=True)


app = FastAPI(
    title="Traffic Density Estimator",
    description=(
        "프로브 차량의 원시 FCD 데이터(300초)를 입력받아 "
        "교통 밀도(veh/km)와 교통량(veh/hr)을 추정합니다.\n\n"
        "**추론 흐름:** FCD → 6채널 trajectory → 피처 추출 → XGBoost → "
        "Underwood FD 베이스라인 + ML 잔차 보정"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:  # type: ignore[return-value]
    """원시 FCD 레코드로부터 교통 밀도와 교통량을 추정합니다.

    - **fcd_records**: 300행의 FCD 레코드 (1초 간격, time/x/y/speed/brake)
    - **speed_limit**: 해당 도로의 제한속도 (m/s)
    - **num_lanes**: 차로 수
    """
    registry: ModelRegistry = request.app.state.registry
    fcd_dicts = [r.model_dump() for r in body.fcd_records]
    try:
        result = predict_density(
            fcd_records=fcd_dicts,
            speed_limit=body.speed_limit,
            num_lanes=body.num_lanes,
            registry=registry,
        )
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Prediction failed: {e}"},
        )

    # Save to database (best-effort — prediction result is returned even on DB failure)
    prediction_id = None
    if getattr(request.app.state, "db_available", False):
        try:
            from src.api.crud import save_prediction
            from src.api.database import async_session_factory

            assert async_session_factory is not None
            async with async_session_factory() as session:
                prediction_id = await save_prediction(
                    session=session,
                    speed_limit=body.speed_limit,
                    num_lanes=body.num_lanes,
                    fcd_records=fcd_dicts,
                    result=result,
                )
        except Exception:
            logger.warning("Failed to save prediction to DB", exc_info=True)

    return PredictResponse(prediction_id=prediction_id, **result)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """서버 상태 및 로드된 모델 정보를 반환합니다."""
    registry: ModelRegistry = request.app.state.registry
    return HealthResponse(
        status="ok",
        model="xgboost",
        model_path=str(registry.model_path),
        residual_correction=registry.residual_enabled,
        n_features=len(registry.feature_columns),
    )
