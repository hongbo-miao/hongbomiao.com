#!/usr/bin/env bash

set -e

purgecss --css build/static/css/*.css --content build/static/index.html build/static/js/*.js --out build/static/css
babel-node tmp/runUpdateHeaders.js
rm -r tmp
