#!/usr/bin/env bash

set -e

yarn run babel scripts --out-dir tmp --extensions '.ts,.tsx'
yarn run babel src/shared/utils/paths.ts --out-file src/shared/utils/paths.js
babel-node tmp/runBuildSitemap.js
