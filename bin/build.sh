#!/usr/bin/env bash
set -e

npm install

cd web
npm install
npm run build

cd ..
cp -r web/build/ api-node/public/

cd api-node
npm install
npm run build
