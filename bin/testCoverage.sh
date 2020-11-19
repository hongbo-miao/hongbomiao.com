#!/usr/bin/env bash

set -e

cd web
yarn test:coverage

cd ../server
yarn test:coverage

cd ..
mkdir -p tmp
cp web/coverage/coverage-final.json tmp/coverage-final-web.json
cp server/coverage/coverage-final.json tmp/coverage-final-server.json

mkdir -p coverage
nyc merge tmp coverage/coverage-final.json
nyc report --temp-dir coverage --report-dir coverage --reporter=text --reporter=lcov

rm -rf tmp
