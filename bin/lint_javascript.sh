#!/usr/bin/env bash
set -e

cd api-node && npm run lint:javascript
cd ../data-visualization/grafana/hm-panel-plugin && npm run lint:javascript
cd ../ethereum && npm run lint:javascript
cd ../../mobile-react-native && npm run lint:javascript
cd ../web && npm run lint:javascript
cd ../web-cypress && npm run lint:javascript
