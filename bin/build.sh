#!/usr/bin/env bash

set -e

cd client
yarn install
yarn build

cd ..
cp -r client/build/ server/dist/

cd server
yarn install
yarn build
