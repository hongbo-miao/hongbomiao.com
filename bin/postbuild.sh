#!/usr/bin/env bash

set -e

babel-node tmp/postbuild.js
purgecss --css build/static/css/*.css --content build/static/index.html build/static/js/*.js --out build/static/css
rm -r tmp
