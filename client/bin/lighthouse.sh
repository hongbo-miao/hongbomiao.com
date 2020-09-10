#!/usr/bin/env bash

set -e

# https://create-react-app.dev/docs/adding-custom-environment-variables/
# In create-react-app, when you run 'yarn build' to make a production bundle, it is always equal to 'production'.
# You cannot override NODE_ENV manually.
# So use development version .env as production version
cp -n .env.development.local.example .env.production.local || true

lhci autorun --upload.target=temporary-public-storage
