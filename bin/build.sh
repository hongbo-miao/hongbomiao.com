#!/usr/bin/env bash

set -e

yarn install

cd client
yarn install
cross-env REACT_APP_LIGHTSTEP_TOKEN="$REACT_APP_LIGHTSTEP_TOKEN" yarn build

cd ..
cp -r client/build/ server/dist/

cd server
yarn install
yarn build
