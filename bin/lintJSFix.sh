#!/usr/bin/env bash
set -e

cd api-node && npm run lint:js:fix
cd ../ethereum && npm run lint:js:fix
cd ../grafana/hm-panel-plugin && npm run lint:js:fix
cd ../../mobile-react-native && npm run lint:js:fix
cd ../web && npm run lint:js:fix
cd ../web-cypress && npm run lint:js:fix
