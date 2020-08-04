#!/usr/bin/env bash

set -e

yarn install

cd client
yarn install
yarn build

cd ..
cp -r client/build/ server/dist/

cd server
yarn install
yarn build
