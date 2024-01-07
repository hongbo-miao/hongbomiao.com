#!/usr/bin/env bash
set -e

# .env
cp -n web/.env.production.local.example web/.env.production.local || true

cp -n api-node/.env.development.local.example api-node/.env.development.local || true
cp -n api-node/.env.production.local.example api-node/.env.production.local || true
cp -n api-node/postgres/.env.development.local.example api-node/postgres/.env.development.local || true
cp -n api-node/postgres/.env.production.local.example api-node/postgres/.env.production.local || true

cp -n data-processing/flink/applications/stream-tweets/src/main/resources/application-development.properties.template data-processing/flink/applications/stream-tweets/src/main/resources/application-development.properties || true
cp -n data-processing/flink/applications/stream-tweets/src/main/resources/application-production.properties.template data-processing/flink/applications/stream-tweets/src/main/resources/application-production.properties || true

# Install dependencies
npm install
cd api-node && npm install
cd ../ethereum && npm install
cd ../grafana/hm-panel-plugin && npm install
cd ../../mobile-react-native && npm install
cd ../web && npm install
cd ../web-cypress && npm install
