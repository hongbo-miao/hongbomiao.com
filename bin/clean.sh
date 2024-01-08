#!/usr/bin/env bash
set -e

# Root
rm -r -f coverage/
rm -r -f node_modules/

# Web
cd web
rm -r -f .lighthouseci/
rm -r -f build/
rm -r -f coverage/
rm -r -f node_modules/
rm -r -f tmp/
rm -f public/sitemap.xml

# Mobile - React Native
cd ../mobile-react-native
rm -r -f .expo/
rm -r -f coverage/
rm -r -f node_modules/

# API - Node.js
cd ../api-node
rm -r -f .clinic/
rm -r -f build/
rm -r -f coverage/
rm -r -f public/
rm -r -f node_modules/

# API - Go
cd ../api-go
rm -r -f web/

# Cypress
cd ../web-cypress
rm -r -f node_modules/

# Ethereum
cd ../ethereum
rm -r -f node_modules/

# Grafana
cd ../data-visualization/grafana/hm-panel-plugin
rm -r -f node_modules/
