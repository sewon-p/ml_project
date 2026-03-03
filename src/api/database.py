"""Async database engine, session factory, and initialization."""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

engine = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(database_url: str) -> None:
    """Create the async engine, tables, and TimescaleDB hypertable."""
    global engine, async_session_factory

    engine = create_async_engine(database_url, echo=False)
    async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

    from src.api.models_db import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create TimescaleDB hypertable (idempotent)
    async with async_session_factory() as session:
        try:
            await session.execute(
                text(
                    "SELECT create_hypertable('fcd_records', 'recorded_at', if_not_exists => TRUE)"
                )
            )
            await session.commit()
            logger.info("TimescaleDB hypertable 'fcd_records' ready")
        except Exception:
            await session.rollback()
            logger.warning(
                "Could not create hypertable (TimescaleDB extension may not be available). "
                "Falling back to regular table."
            )

    logger.info("Database initialized: %s", database_url.split("@")[-1])


async def close_db() -> None:
    """Dispose the engine connection pool."""
    global engine, async_session_factory
    if engine is not None:
        await engine.dispose()
        engine = None
        async_session_factory = None


async def get_session() -> AsyncSession:
    """Yield an async session (for use with FastAPI Depends)."""
    if async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with async_session_factory() as session:
        yield session
