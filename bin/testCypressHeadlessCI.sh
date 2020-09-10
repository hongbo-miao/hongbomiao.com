#!/usr/bin/env bash

set -e

# https://create-react-app.dev/docs/adding-custom-environment-variables/
# In create-react-app, when you run 'yarn build' to make a production bundle, it is always equal to 'production'.
# You cannot override NODE_ENV manually.
# So use development version .env as production version
cp -n client/.env.development.local.example client/.env.production.local || true

cp -n server/private/ssl/hongbomiao.crt.example server/private/ssl/hongbomiao.crt || true
cp -n server/private/ssl/hongbomiao.key.example server/private/ssl/hongbomiao.key || true

START_SERVER_AND_TEST_INSECURE=1 start-server-and-test 'cd server && yarn dev:cypress' https://localhost:5000 './node_modules/.bin/cypress run --config-file ./cypress/cypress.ci.json'
