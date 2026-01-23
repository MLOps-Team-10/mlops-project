#!/usr/bin/env bash

# Important
# remember to download your service account key and save
# it in the root directory as `service_account_key.json`

docker build -t  my_fastapi_app -f dockerfiles/api.dockerfile .