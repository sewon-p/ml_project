FROM python:3.13-slim AS base

WORKDIR /app
ENV PYTHONPATH=/app

# System deps (numpy/scipy build 시 필요할 수 있음)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 (캐시 활용)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[api,streaming,gcp]"

# 소스 코드 복사
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# 모바일 웹페이지 정적 파일
COPY static/ static/

# GIS link layer for mobile GPS -> road link matching
COPY data/gis/ data/gis/

# 모델 + 피처 스키마 복사 (런타임에 필요)
# docker build 시 이 파일들이 존재해야 함
COPY outputs_xgboost/xgboost_best.pkl outputs_xgboost/xgboost_best.pkl
COPY data/features/dataset.parquet data/features/dataset.parquet

ENV CONFIG_PATH=configs/default.yaml

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
