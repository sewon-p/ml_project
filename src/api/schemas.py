"""Pydantic request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FCDRecord(BaseModel):
    """Single FCD (Floating Car Data) record — 1초 단위 프로브 차량 관측."""

    time: float = Field(description="시각 (초)")
    x: float = Field(description="종방향 위치 (m)")
    y: float = Field(description="횡방향 위치 (m)")
    speed: float = Field(description="순간 속도 (m/s)")
    brake: float = Field(default=0.0, description="브레이크 신호 (0 또는 1)")


class PredictRequest(BaseModel):
    """POST /predict 요청 — 300초 프로브 차량 FCD + 도로 조건."""

    fcd_records: list[FCDRecord] = Field(
        description="프로브 차량 FCD 레코드 (300행, 1초 간격)",
    )
    speed_limit: float = Field(
        description="제한속도 (m/s). 예: 60km/h → 16.67, 80km/h → 22.22",
    )
    num_lanes: int = Field(
        description="차로 수 (1~3)",
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
    """POST /predict 응답 — 교통 밀도/교통량 추정 결과."""

    prediction_id: int | None = Field(
        default=None, description="DB 저장된 prediction ID (DB 미연결 시 null)"
    )
    density: float = Field(description="최종 밀도 (veh/km) = FD 베이스라인 + ML 보정")
    flow: float = Field(description="최종 교통량 (veh/hr)")
    fd_density: float = Field(description="Underwood FD 베이스라인 밀도 (veh/km)")
    fd_flow: float = Field(description="Underwood FD 베이스라인 교통량 (veh/hr)")
    residual_density: float = Field(description="ML 보정값 Δk (veh/km)")

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


class HealthResponse(BaseModel):
    """GET /health 응답."""

    status: str = Field(description="서버 상태")
    model: str = Field(description="모델 타입")
    model_path: str = Field(description="모델 파일 경로")
    residual_correction: bool = Field(description="잔차 보정 사용 여부")
    n_features: int = Field(description="피처 수")
