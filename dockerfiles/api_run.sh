#!/usr/bin/env bash

# Important
# remember to download your service account key and save
# it in the root directory as `service_account_key.json`

docker run --rm \
  --name my_fastapi_cnt \
  -p 80:80 \
  -v "$(pwd)/service_account_key.json":/app/service_account_key.json \
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/service_account_key.json" \
  my_fastapi_app