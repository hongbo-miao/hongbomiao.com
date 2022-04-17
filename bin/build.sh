#!/usr/bin/env bash
set -e

npm install

cd web
npm install
rm -f ./.eslintrc.js # Skip lint check during react-scripts build
npm run build

cd ..
cp -r web/build/ api-node/public/

cd api-node
npm install
npm run build
