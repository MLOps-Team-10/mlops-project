#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="train:latest"
#syncing uv
uv sync
# build
docker build -f dockerfiles/train.dockerfile -t "${IMAGE_NAME}" .

# run 
docker run --rm \
  --name train \
  --shm-size=2g \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD/.secrets/bucket_manager.json:/run/secrets/gcp-sa.json" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp-sa.json \
  "${IMAGE_NAME}"
