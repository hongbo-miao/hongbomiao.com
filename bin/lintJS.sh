#!/usr/bin/env bash
set -e

cd api-node && npm run lint:js
cd ../ethereum && npm run lint:js
cd ../grafana/hm-panel-plugin && npm run lint:js
cd ../../mobile-react-native && npm run lint:js
cd ../web && npm run lint:js
cd ../web-cypress && npm run lint:js
