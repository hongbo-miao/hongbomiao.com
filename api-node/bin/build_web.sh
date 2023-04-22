#!/usr/bin/env bash
set -e

cd ../web
npm install
npm run build

cd ..
cp -r web/build/ api-node/public/
