#!/usr/bin/env bash
set -euo pipefail


SERVICE_ACCOUNT="bucket-manager@mlops-exercises-484210.iam.gserviceaccount.com"
REGION="europe-west1"
CONFIG="config_cpu.yaml"

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --config="${CONFIG}" \
  --service-account="${SERVICE_ACCOUNT}" \
  --display-name="No_cloud_run"