#!/usr/bin/env bash
set -e

cd ../web
npm install
npm run build

cd ..
cp -R web/dist/ api-node/public/
