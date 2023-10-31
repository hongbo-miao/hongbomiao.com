#!/bin/sh
set -e

# purgecss --css build/static/css/*.css --content build/index.html build/static/js/*.js --output build/static/css
rm -r -f tmp/
