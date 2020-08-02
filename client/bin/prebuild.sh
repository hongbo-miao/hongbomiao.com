#!/usr/bin/env bash

set -e

babel scripts --out-dir tmp --extensions '.ts,.tsx'
babel src/shared/utils/paths.ts --out-file src/shared/utils/paths.js
node tmp/runBuildSitemap.js
rm src/shared/utils/paths.js
