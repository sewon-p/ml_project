FROM python:3.13-slim AS base

WORKDIR /app
ENV PYTHONPATH=/app

# System deps (may be needed for numpy/scipy build)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[api,streaming,gcp]"

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Mobile web static files
COPY static/ static/

# GIS link layer for mobile GPS -> road link matching
COPY data/gis/ data/gis/

# Copy model weights + feature schema (required at runtime)
# These files must exist at docker build time
COPY outputs_xgboost/xgboost_best.pkl outputs_xgboost/xgboost_best.pkl
COPY data/features/dataset.parquet data/features/dataset.parquet

ENV CONFIG_PATH=configs/default.yaml

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
