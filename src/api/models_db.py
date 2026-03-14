"""SQLAlchemy ORM models for traffic prediction storage."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    speed_limit: Mapped[float] = mapped_column(Float, nullable=False)
    num_lanes: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    predictions: Mapped[list[Prediction]] = relationship(back_populates="scenario")


class RoadLink(Base):
    __tablename__ = "road_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    link_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    road_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    geometry_geojson: Mapped[str | None] = mapped_column(Text, nullable=True)
    center_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    center_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    predictions: Mapped[list[Prediction]] = relationship(back_populates="road_link")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id"), nullable=False)
    road_link_id: Mapped[int | None] = mapped_column(ForeignKey("road_links.id"), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    density: Mapped[float] = mapped_column(Float, nullable=False)
    flow: Mapped[float] = mapped_column(Float, nullable=False)
    fd_density: Mapped[float] = mapped_column(Float, nullable=False)
    fd_flow: Mapped[float] = mapped_column(Float, nullable=False)
    residual_density: Mapped[float] = mapped_column(Float, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    scenario: Mapped[Scenario] = relationship(back_populates="predictions")
    road_link: Mapped[RoadLink | None] = relationship(back_populates="predictions")
    fcd_records: Mapped[list[FCDRecordRow]] = relationship(back_populates="prediction")


class FCDRecordRow(Base):
    __tablename__ = "fcd_records"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    time: Mapped[float] = mapped_column(Float, nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    speed: Mapped[float] = mapped_column(Float, nullable=False)
    brake: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    prediction: Mapped[Prediction] = relationship(back_populates="fcd_records")
