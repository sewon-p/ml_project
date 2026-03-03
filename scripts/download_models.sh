#!/usr/bin/env bash
# Download model artifacts from the latest GitHub release.
# Usage: bash scripts/download_models.sh [TAG]
#   TAG  - release tag (default: latest)
#
# Requires: gh CLI (https://cli.github.com/) authenticated.

set -euo pipefail

REPO="ParkSewon-PM/ml_project"
TAG="${1:-latest}"

echo "==> Downloading model artifacts from release: ${TAG}"

mkdir -p outputs_xgboost data/features

if [ "$TAG" = "latest" ]; then
    gh release download --repo "$REPO" --pattern "xgboost_best.pkl" --dir outputs_xgboost --clobber
    gh release download --repo "$REPO" --pattern "dataset.parquet"  --dir data/features    --clobber
else
    gh release download "$TAG" --repo "$REPO" --pattern "xgboost_best.pkl" --dir outputs_xgboost --clobber
    gh release download "$TAG" --repo "$REPO" --pattern "dataset.parquet"  --dir data/features    --clobber
fi

echo "==> Done. Files downloaded:"
ls -lh outputs_xgboost/xgboost_best.pkl data/features/dataset.parquet
