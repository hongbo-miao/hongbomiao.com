#!/usr/bin/env bash

set -e

yarn run babel bin --out-dir tmp --extensions '.ts,.tsx'
babel-node tmp/runBuildSitemap.js
