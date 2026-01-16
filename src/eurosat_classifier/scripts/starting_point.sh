#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="train:latest"
#syncing uv
uv sync
# build
docker build -f dockerfiles/train.dockerfile -t "${IMAGE_NAME}" .

# run (mount data folder + shm)
docker run --rm \
  --name train \
  --shm-size=2g \
  --env-file .env \
  -v "$(pwd)/data:/app/data" \
  "${IMAGE_NAME}"
