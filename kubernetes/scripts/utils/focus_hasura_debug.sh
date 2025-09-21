#!/usr/bin/env bash
set -e

echo "# Focus Hasura Debug"
source kubernetes/bin/utils/install_postgres.sh
source kubernetes/bin/utils/install_timescaledb.sh
source kubernetes/bin/utils/install_dgraph.sh
source kubernetes/bin/utils/install_graphql_server_without_opal.sh
echo "=================================================="
