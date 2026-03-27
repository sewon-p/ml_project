"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FCDRecord(BaseModel):
    """Single Floating Car Data record."""

    time: float = Field(description="Timestamp in seconds")
    x: float = Field(description="Longitudinal position in meters")
    y: float = Field(description="Lateral position in meters")
    speed: float = Field(description="Instantaneous speed in m/s")
    brake: float = Field(default=0.0, description="Brake flag, 0 or 1")


class PredictRequest(BaseModel):
    """Payload for POST /predict."""

    fcd_records: list[FCDRecord] = Field(
        description="Probe-vehicle FCD records, typically 300 rows at 1 Hz",
    )
    speed_limit: float = Field(
        description="Speed limit in m/s, for example 16.67 for 60 km/h",
    )
    num_lanes: int = Field(
        description="Lane count",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fcd_records": [
                        {"time": 0.0, "x": 0.0, "y": 0.1, "speed": 15.2, "brake": 0},
                        {"time": 1.0, "x": 15.3, "y": 0.1, "speed": 15.3, "brake": 0},
                        {"time": 2.0, "x": 30.5, "y": 0.2, "speed": 15.1, "brake": 0},
                    ],
                    "speed_limit": 22.22,
                    "num_lanes": 2,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Prediction output returned by POST /predict."""

    prediction_id: int | None = Field(
        default=None, description="Stored prediction ID, or null when DB storage is unavailable"
    )
    density: float = Field(description="Final density estimate in veh/km")
    flow: float = Field(description="Final flow estimate in veh/h")
    fd_density: float = Field(description="Underwood FD baseline density in veh/km")
    fd_flow: float = Field(description="Underwood FD baseline flow in veh/h")
    residual_density: float = Field(description="ML residual correction Delta k")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction_id": 1,
                    "density": 12.34,
                    "flow": 456.78,
                    "fd_density": 8.5,
                    "fd_flow": 320.1,
                    "residual_density": 3.84,
                }
            ]
        }
    }


class RoadLinkSummary(BaseModel):
    """Road link metadata exposed to the map GUI."""

    link_id: str = Field(description="External GIS link identifier")
    road_name: str | None = Field(default=None, description="Human-readable road name")
    source: str = Field(description="Source of the link metadata")
    geometry_geojson: str | None = Field(default=None, description="GeoJSON geometry string")
    center_lat: float | None = Field(default=None, description="Center latitude")
    center_lon: float | None = Field(default=None, description="Center longitude")
    road_rank: str | None = Field(default=None, description="MOCT road hierarchy rank")
    link_length_m: float | None = Field(default=None, description="Link length in meters")
    lanes: int | None = Field(default=None, description="Number of lanes")
    max_spd: float | None = Field(default=None, description="Max speed km/h")


class LinkPredictionSummary(BaseModel):
    """Latest or historical prediction tied to a road link."""

    prediction_id: int = Field(description="Prediction identifier")
    session_id: str | None = Field(default=None, description="Mobile session identifier")
    observed_at: datetime = Field(description="Timestamp of the prediction")
    density: float = Field(description="Estimated density in veh/km")
    flow: float = Field(description="Estimated flow in veh/h")
    fd_density: float = Field(description="FD baseline density in veh/km")
    fd_flow: float = Field(description="FD baseline flow in veh/h")
    residual_density: float = Field(description="Residual correction Delta k")


class LinkLatestResponse(BaseModel):
    """Map layer payload: one latest prediction per link."""

    link: RoadLinkSummary
    latest_prediction: LinkPredictionSummary


class LinkHistoryResponse(BaseModel):
    """Historical predictions for one link."""

    link: RoadLinkSummary
    history: list[LinkPredictionSummary]


class PredictionDetailResponse(BaseModel):
    """Detailed prediction view for the right-side drawer/modal."""

    prediction_id: int = Field(description="Prediction identifier")
    link: RoadLinkSummary | None = Field(default=None, description="Linked road segment")
    session_id: str | None = Field(default=None, description="Mobile session identifier")
    observed_at: datetime = Field(description="Prediction timestamp")
    density: float = Field(description="Estimated density in veh/km")
    flow: float = Field(description="Estimated flow in veh/h")
    fd_density: float = Field(description="FD baseline density in veh/km")
    fd_flow: float = Field(description="FD baseline flow in veh/h")
    residual_density: float = Field(description="Residual correction Delta k")
    fcd_records: list[FCDRecord] = Field(description="Underlying 300-second FCD window")


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = Field(description="Service status")
    model: str = Field(description="Loaded model type")
    model_path: str = Field(description="Resolved model file path")
    residual_correction: bool = Field(description="Whether residual correction is enabled")
    n_features: int = Field(description="Feature count used by the model")
