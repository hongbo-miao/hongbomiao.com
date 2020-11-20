#!/usr/bin/env bash

set -e

# https://create-react-app.dev/docs/adding-custom-environment-variables/
# In create-react-app, when you run 'yarn build' to make a production bundle, it is always equal to 'production'.
# So using REACT_APP_SERVER_DOMAIN=localhost and REACT_APP_SERVER_PORT=5000 in .env.production.local.example
# to avoid sending data to production server
cp -n .env.production.local.example .env.production.local || true

lhci autorun --upload.target=temporary-public-storage
