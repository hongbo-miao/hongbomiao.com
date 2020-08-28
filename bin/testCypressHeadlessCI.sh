#!/usr/bin/env bash

set -e

cp -n server/private/ssl/hongbomiao.crt.example server/private/ssl/hongbomiao.crt || true
cp -n server/private/ssl/hongbomiao.key.example server/private/ssl/hongbomiao.key || true

START_SERVER_AND_TEST_INSECURE=1 start-server-and-test 'cd server && yarn dev' https://localhost:5000 'pwd && ./node_modules/.bin/cypress run --config-file ./cypress/cypress.ci.json'
