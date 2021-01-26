#!/usr/bin/env bash

set -e

yarn install

cd web
yarn install
cross-env REACT_APP_LIGHTSTEP_TOKEN="$REACT_APP_LIGHTSTEP_TOKEN" yarn build

cd ..
cp -r web/build/ api/dist/

cd api
yarn install
yarn build
