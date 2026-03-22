#!/usr/bin/env bash
# Download deployment assets from GitHub releases.
# Usage: bash scripts/download_models.sh [TAG]
#   TAG  - asset bundle release tag override (default: deploy/asset_bundle.env)
#
# Requires: gh CLI (https://cli.github.com/) authenticated.

set -euo pipefail

REPO="ParkSewon-PM/ml_project"
source deploy/asset_bundle.env
TAG="${1:-$ASSET_RELEASE_TAG}"

echo "==> Downloading deployment assets from release: ${TAG}"

mkdir -p outputs_xgboost data/gis

gh release download "$TAG" --repo "$REPO" --pattern "$MODEL_ASSET" --dir outputs_xgboost --clobber
gh release download "$TAG" --repo "$REPO" --pattern "$GIS_ASSET" --dir data/gis --clobber

echo "==> Done. Files downloaded:"
ls -lh outputs_xgboost/xgboost_best.pkl data/gis/seoul_links.geojson
