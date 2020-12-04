#!/usr/bin/env bash

set -e

# .env
cp -n web/.env.development.local.example web/.env.development.local || true
cp -n web/.env.production.local.example web/.env.production.local || true

cp -n server/.env.development.local.example server/.env.development.local || true
cp -n server/.env.production.local.example server/.env.production.local || true

cp -n docker/postgres/.env.development.local.example docker/postgres/.env.development.local || true
cp -n docker/postgres/.env.production.local.example docker/postgres/.env.production.local || true

# Install dependencies
yarn install
cd web && yarn install
cd mobile && yarn install
cd ../server && yarn install
