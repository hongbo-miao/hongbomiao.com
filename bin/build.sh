#!/usr/bin/env bash

set -e

npm install

cd web
npm install
npm run build

cd ..
cp -r web/build/ api/public/
cp -r web/build/ api-go/web/

cd api
npm install
npm run build
