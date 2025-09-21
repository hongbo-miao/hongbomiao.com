#!/bin/sh
set -e

rm -r -f tmp
swc scripts --out-dir=tmp --extensions='.ts,.tsx'
swc src/shared/utils/paths.ts --out-file=tmp/src/shared/utils/paths.js
node tmp/scripts/runBuildSitemap.js
rm -r -f tmp
