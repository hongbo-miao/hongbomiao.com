#!/usr/bin/env bash
set -e

# .env
cp -n web/.env.production.local.example web/.env.production.local || true

cp -n api-node/.env.development.local.example api-node/.env.development.local || true
cp -n api-node/.env.production.local.example api-node/.env.production.local || true

cp -n streaming/src/main/resources/application-development.properties.template streaming/src/main/resources/application-development.properties || true
cp -n streaming/src/main/resources/application-production.properties.template streaming/src/main/resources/application-production.properties || true

cp -n docker/postgres/.env.development.local.example docker/postgres/.env.development.local || true
cp -n docker/postgres/.env.production.local.example docker/postgres/.env.production.local || true

# Install dependencies
npm install
cd api-node && npm install
cd ../ethereum && npm install
cd ../grafana/hm-panel-plugin && npm install
cd ../mobile-react-native && npm install
cd ../web && npm install
cd ../web-cypress && npm install
