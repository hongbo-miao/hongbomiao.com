#!/usr/bin/env bash
set -e

echo "# Focus Hasura Debug"
source kubernetes/bin/utils/installPostgres.sh
source kubernetes/bin/utils/installTimescaleDB.sh
source kubernetes/bin/utils/installDgraph.sh
source kubernetes/bin/utils/installGraphQLServerWithoutOPAL.sh
echo "=================================================="
