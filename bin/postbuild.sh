#!/usr/bin/env bash

set -e

rm build/static/js/*.map
rm build/static/css/*.map
babel-node tmp/postbuild.js
