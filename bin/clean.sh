#!/usr/bin/env bash

set -e

# Root
rm -rf coverage
rm -rf node_modules

# Web
cd web
rm -rf build
rm -rf tmp
rm -rf coverage
rm -rf node_modules
rm -f public/sitemap.xml

# Mobile
cd ../mobile
rm -rf node_modules

# Server
cd ../server
rm -rf .clinic
rm -rf build
rm -rf dist
rm -rf coverage
rm -rf node_modules
