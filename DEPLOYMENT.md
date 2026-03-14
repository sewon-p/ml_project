# Deployment

## Service Layout

- `8000 /dashboard`: UrbanFlow Console
- `8000 /map`: link map
- `8000 /mobile`: mobile ingest UI
- `8000 /ml-pipeline/`: ML Pipeline
- `8000 /docs`: FastAPI docs

## Local Start

```bash
python scripts/run_console.py
```

## GitHub Actions

### CI

Triggered on:

- push to `main`
- pull request to `main`

Checks:

- Ruff
- mypy on `src/api`
- pytest
- Docker build
- container import smoke check

### CD

Triggered on:

- GitHub release publish
- manual `workflow_dispatch`

Steps:

1. Download release model artifacts
2. Build container image
3. Push image to GHCR and Artifact Registry
4. Deploy to Cloud Run
5. Verify `/health`, `/dashboard`, `/ml-pipeline/`

## Required GitHub Secrets

- `GCP_SA_KEY`

## Required Release Assets

- `xgboost_best.pkl`
- `dataset.parquet`
- `seoul_links.geojson`

## Cloud Run Notes

- The image must contain `src/`, `scripts/`, `configs/`, and `static/`
- `PYTHONPATH=/app` is set in the Docker image
- The app serves both the console and the ML Pipeline from one Cloud Run service

## Post-Deploy URLs

- `https://<service-url>/dashboard`
- `https://<service-url>/map`
- `https://<service-url>/mobile`
- `https://<service-url>/ml-pipeline/`
- `https://<service-url>/docs`
