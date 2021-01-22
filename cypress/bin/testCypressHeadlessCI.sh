#!/usr/bin/env bash

set -e

cd ..

# https://create-react-app.dev/docs/adding-custom-environment-variables/
# In create-react-app, when you run 'yarn build' to make a production bundle, it is always equal to 'production'.
# So using REACT_APP_SERVER_DOMAIN=localhost and REACT_APP_SERVER_PORT=5000 in .env.production.local.example
# to avoid sending data to production server
cp -n web/.env.production.local.example web/.env.production.local || true

start-server-and-test 'cd server && yarn dev:cypress' http://localhost:5000 'cd cypress && cypress run --config-file ./cypress.ci.json'
