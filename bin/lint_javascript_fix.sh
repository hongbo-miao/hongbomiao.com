#!/usr/bin/env bash
set -e

cd api-node && npm run lint:javascript:fix
cd ../data-visualization/grafana/hm-panel-plugin && npm run lint:javascript:fix
cd ../ethereum && npm run lint:javascript:fix
cd ../../mobile-react-native && npm run lint:javascript:fix
cd ../web && npm run lint:javascript:fix
cd ../web-cypress && npm run lint:javascript:fix
