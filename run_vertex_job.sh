#!/usr/bin/env bash
set -euo pipefail


gcloud builds submit --config=vertex_ai_train.yaml --ignore-file=.deployignore