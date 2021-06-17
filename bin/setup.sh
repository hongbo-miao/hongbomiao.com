#!/usr/bin/env bash

set -e

# .env
cp -n web/.env.production.local.example web/.env.production.local || true

cp -n api/.env.development.local.example api/.env.development.local || true
cp -n api/.env.production.local.example api/.env.production.local || true

cp -n docker/postgres/.env.development.local.example docker/postgres/.env.development.local || true
cp -n docker/postgres/.env.production.local.example docker/postgres/.env.production.local || true

# Install dependencies
npm install
npx husky install
cd web && npm install
cd ../mobile && npm install
cd ../api && npm install
cd ../cypress && npm install
cd ../ethereum && npm install
