#!/usr/bin/env bash

set -e

# SSL
cp -n web/private/ssl/hongbomiao.crt.example web/private/ssl/hongbomiao.crt || true
cp -n web/private/ssl/hongbomiao.key.example web/private/ssl/hongbomiao.key || true

cp -n server/private/ssl/hongbomiao.crt.example server/private/ssl/hongbomiao.crt || true
cp -n server/private/ssl/hongbomiao.key.example server/private/ssl/hongbomiao.key || true

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
