#!/usr/bin/env bash
set -euo pipefail


gcloud builds submit . --config=cloud_deploy/vertex_ai_train.yaml --ignore-file=.deployignore
