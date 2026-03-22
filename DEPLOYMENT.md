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

1. Resolve the pinned asset bundle tag from `deploy/asset_bundle.env`
2. Download deployable assets from that asset bundle release
3. Build container image
4. Push image to GHCR and Artifact Registry
5. Deploy to Cloud Run
6. Verify `/health`, `/dashboard`, `/ml-pipeline/`

### Why This Structure

- Code releases and deployable assets are intentionally decoupled.
- Normal code-only releases do not need model assets re-uploaded.
- Only model/GIS changes require updating `deploy/asset_bundle.env`.
- This avoids brittle "search previous releases for missing assets" logic during CD.

### When Updating The Model

1. Publish a release that contains the new deployable assets.
2. Update `deploy/asset_bundle.env` to point to that asset release tag.
3. Merge and deploy the code release that should use the new bundle.

## Required Asset Bundle Release Contents

- `xgboost_best.pkl`
- `seoul_links.geojson`

## Runtime Metadata In Repo

- `configs/runtime/feature_columns.json`

The runtime API no longer depends on downloading `dataset.parquet` just to recover feature column order.

## Previous Notes

Old flow:

1. Download release model artifacts
2. Build container image
3. Push image to GHCR and Artifact Registry
4. Deploy to Cloud Run
5. Verify `/health`, `/dashboard`, `/ml-pipeline/`

## Required GitHub Secrets

- `GCP_SA_KEY`

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
