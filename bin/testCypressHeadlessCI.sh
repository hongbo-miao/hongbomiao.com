#!/usr/bin/env bash

set -e

# https://create-react-app.dev/docs/adding-custom-environment-variables/
# In create-react-app, when you run 'yarn build' to make a production bundle, it is always equal to 'production'.
# So using REACT_APP_SERVER_DOMAIN=localhost and REACT_APP_SERVER_PORT=5000 in .env.production.local.example
# to avoid sending data to production server
cp -n client/.env.production.local.example client/.env.production.local || true

cp -n server/private/ssl/hongbomiao.crt.example server/private/ssl/hongbomiao.crt || true
cp -n server/private/ssl/hongbomiao.key.example server/private/ssl/hongbomiao.key || true

START_SERVER_AND_TEST_INSECURE=1 start-server-and-test 'cd server && yarn dev:cypress' https://localhost:5000 'cypress run --config-file ./cypress/cypress.ci.json'
