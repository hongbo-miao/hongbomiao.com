#!/usr/bin/env bash

set -e

cd web
yarn test:coverage

cd ../api
yarn test:coverage

cd ..
mkdir -p tmp
cp web/coverage/coverage-final.json tmp/coverage-final-web.json
cp api/coverage/coverage-final.json tmp/coverage-final-api.json

mkdir -p coverage
nyc merge tmp coverage/coverage-final.json
nyc report --temp-dir coverage --report-dir coverage --reporter=text --reporter=lcov

rm -rf tmp
