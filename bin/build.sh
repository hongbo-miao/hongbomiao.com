#!/usr/bin/env bash

set -e

npm install

cd web
npm install
cross-env REACT_APP_LIGHTSTEP_TOKEN="$REACT_APP_LIGHTSTEP_TOKEN" npm run build

cd ..
cp -r web/build/ api/public/

cd api
npm install
npm run build
