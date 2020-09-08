#!/usr/bin/env bash

set -e

# .env
cp -n .env.development.local.example .env.development.local || true

cp -n client/.env.development.local.example client/.env.development.local || true
cp -n client/.env.production.local.example client/.env.production.local || true

cp -n server/.env.development.local.example server/.env.development.local || true
cp -n server/.env.production.local.example server/.env.production.local || true

# SSL
cp -n client/private/ssl/hongbomiao.crt.example client/private/ssl/hongbomiao.crt || true
cp -n client/private/ssl/hongbomiao.key.example client/private/ssl/hongbomiao.key || true

cp -n server/private/ssl/hongbomiao.crt.example server/private/ssl/hongbomiao.crt || true
cp -n server/private/ssl/hongbomiao.key.example server/private/ssl/hongbomiao.key || true

# Install
yarn install
cd client && yarn install
cd ../server && yarn install
