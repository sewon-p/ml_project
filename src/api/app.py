"""FastAPI application for traffic density estimation."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from src.api.dependencies import ModelRegistry
from src.api.inference import predict_density
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse

try:
    from fastapi.staticfiles import StaticFiles

    _HAS_STATICFILES = True
except ImportError:
    _HAS_STATICFILES = False

logger = logging.getLogger(__name__)


class _PrefixRestoringApp:
    """ASGI wrapper that restores a stripped mount prefix for a sub-application."""

    def __init__(self, asgi_app, prefix: str) -> None:
        self.asgi_app = asgi_app
        self.prefix = prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"}:
            scope = dict(scope)
            scope["path"] = f"{self.prefix}{scope['path']}"
        await self.asgi_app(scope, receive, send)


def _get_database_url() -> str | None:
    """Resolve DATABASE_URL from environment or config."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")
    try:
        import yaml  # type: ignore[import-untyped]

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("database", {}).get("url")
    except Exception:
        return None


def _ml_pipeline_mutations_disabled() -> bool:
    """Disable expensive ML Pipeline mutations on hosted environments."""
    if os.environ.get("DISABLE_ML_PIPELINE_MUTATIONS") == "1":
        return True
    return bool(os.environ.get("K_SERVICE"))


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

    # Shutdown: close Kafka producer
    try:
        from src.streaming.producer import close as close_kafka

        close_kafka()
    except Exception:
        pass

    # Shutdown: close database connection pool
    if app.state.db_available:
        try:
            from src.api.database import close_db

            await close_db()
        except Exception:
            logger.warning("Error closing database", exc_info=True)


app = FastAPI(
    title="UrbanFlow API",
    description=(
        "Estimate traffic density and flow from a single mobile probe trajectory.\n\n"
        "Inference flow: raw FCD -> 6-channel trajectory -> feature extraction -> "
        "XGBoost residual model -> Underwood FD baseline restoration."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Estimate traffic density and flow from raw FCD records.

    - **fcd_records**: 300 rows of FCD records (1-second interval, time/x/y/speed/brake)
    - **speed_limit**: Road speed limit (m/s)
    - **num_lanes**: Number of lanes
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
        return JSONResponse(  # type: ignore[return-value]
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


# ---------------------------------------------------------------------------
# Streaming / ingest router (graceful — works without kafka-python)
# ---------------------------------------------------------------------------
try:
    from src.api.ingest import router as ingest_router

    app.include_router(ingest_router)
    logger.info("Streaming ingest router mounted (/ingest, /ws/dashboard)")
except Exception:
    logger.info("Streaming ingest router not available — skipping")

# ---------------------------------------------------------------------------
# Map router
# ---------------------------------------------------------------------------
try:
    from src.api.map import router as map_router

    app.include_router(map_router)
    logger.info("Map router mounted (/map/*)")
except Exception:
    logger.info("Map router not available — skipping")

# ---------------------------------------------------------------------------
# ML Pipeline sub-app
# ---------------------------------------------------------------------------
try:
    from scripts.dashboard import app as ml_pipeline_app

    if _ml_pipeline_mutations_disabled():
        blocked_detail = "ML Pipeline execution is disabled on cloud"

        @app.post("/api/run", include_in_schema=False)
        async def block_pipeline_run() -> None:
            raise HTTPException(status_code=403, detail=blocked_detail)

        @app.post("/api/cancel", include_in_schema=False)
        async def block_pipeline_cancel() -> None:
            raise HTTPException(status_code=403, detail=blocked_detail)

        @app.post("/api/force-reset", include_in_schema=False)
        async def block_pipeline_force_reset() -> None:
            raise HTTPException(status_code=403, detail=blocked_detail)

        @app.post("/api/clean", include_in_schema=False)
        async def block_pipeline_clean() -> None:
            raise HTTPException(status_code=403, detail=blocked_detail)

        @app.delete("/api/runs/{run_id}", include_in_schema=False)
        async def block_pipeline_delete(run_id: str) -> None:
            del run_id
            raise HTTPException(status_code=403, detail=blocked_detail)

    app.mount("/ml-pipeline", ml_pipeline_app)
    app.mount("/api", _PrefixRestoringApp(ml_pipeline_app, "/api"))
    logger.info("ML Pipeline mounted (/ml-pipeline/*)")
except Exception:
    logger.info("ML Pipeline app not available — skipping", exc_info=True)

# ---------------------------------------------------------------------------
# Static files (mobile.html)
# ---------------------------------------------------------------------------
if _HAS_STATICFILES:
    _static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if _static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
        logger.info("Static files mounted from %s", _static_dir)


def _static_page(name: str) -> FileResponse:
    assert _HAS_STATICFILES
    page = _static_dir / name
    return FileResponse(page)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", include_in_schema=False)
async def dashboard_page() -> FileResponse:
    return _static_page("console-dashboard.html")


@app.get("/ml-pipeline", include_in_schema=False)
async def ml_pipeline_root() -> RedirectResponse:
    return RedirectResponse(url="/ml-pipeline/")


@app.get("/map", include_in_schema=False)
async def map_page() -> FileResponse:
    return _static_page("link-history-map.html")


@app.get("/mobile", include_in_schema=False)
async def mobile_page() -> FileResponse:
    return _static_page("mobile.html")


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return service health and loaded model metadata."""
    if request.query_params.get("format") != "json":
        accept = request.headers.get("accept", "")
        if "text/html" in accept and _HAS_STATICFILES:
            return _static_page("health.html")  # type: ignore[return-value]
    registry: ModelRegistry = request.app.state.registry
    return HealthResponse(
        status="ok",
        model="xgboost",
        model_path=str(registry.model_path),
        residual_correction=registry.residual_enabled,
        n_features=len(registry.feature_columns),
    )
