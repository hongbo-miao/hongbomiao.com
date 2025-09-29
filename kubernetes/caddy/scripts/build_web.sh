#!/usr/bin/env bash
set -e

(cd ../../web && npm install && npm run build)
cp -R ../../web/dist/ public/
