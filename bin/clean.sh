#!/usr/bin/env bash

set -e

rm -rf coverage
rm -rf node_modules

cd client
rm -rf build
rm -rf coverage
rm -rf node_modules
rm -f public/sitemap.xml

cd ../server
rm -rf build
rm -rf coverage
rm -rf dist
rm -rf node_modules
