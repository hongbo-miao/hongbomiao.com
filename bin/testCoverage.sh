#!/usr/bin/env bash

set -e

cd client
yarn test:coverage

cd ../server
yarn test:coverage

cd ..
mkdir -p tmp
cp client/coverage/coverage-final.json tmp/coverage-final-client.json
cp server/coverage/coverage-final.json tmp/coverage-final-server.json

mkdir -p coverage
nyc merge tmp coverage/coverage-final.json
nyc report --temp-dir coverage --report-dir coverage --reporter=text --reporter=lcov

rm -rf tmp
