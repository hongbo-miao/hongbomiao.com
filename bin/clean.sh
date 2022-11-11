#!/usr/bin/env bash
set -e

# Root
rm -rf coverage
rm -rf node_modules

# Web
cd web
rm -rf .lighthouseci
rm -rf build
rm -rf coverage
rm -rf node_modules
rm -rf tmp
rm -f public/sitemap.xml

# Mobile
cd ../mobile
rm -rf .expo
rm -rf coverage
rm -rf node_modules

# API (Node.js)
cd ../api-node
rm -rf .clinic
rm -rf build
rm -rf coverage
rm -rf public
rm -rf node_modules

# API (Go)
cd ../api-go
rm -rf web

# Cypress
cd ../cypress
rm -rf node_modules

# Ethereum
cd ../ethereum
rm -rf node_modules

# Grafana
cd ../grafana/hm-panel-plugin
rm -rf node_modules
