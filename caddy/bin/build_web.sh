#!/usr/bin/env bash
set -e

cd ../web
npm install
npm run build

cd ..
cp -R web/build/ caddy/public/
