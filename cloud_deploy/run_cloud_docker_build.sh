#!/usr/bin/env bash
set -euo pipefail


gcloud builds submit . --config=cloud_deploy/cloudbuild.yaml 