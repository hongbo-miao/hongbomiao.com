#!/usr/bin/env bash
set -e

echo "# Install ClickHouse"
curl --silent https://raw.githubusercontent.com/Altinity/clickhouse-operator/master/deploy/operator-web-installer/clickhouse-operator-install.sh | OPERATOR_NAMESPACE=hm-clickhouse bash
kubectl apply --namespace=hm-clickhouse --filename=https://raw.githubusercontent.com/Altinity/clickhouse-operator/master/docs/chi-examples/01-simple-layout-01-1shard-1repl.yaml
echo "=================================================="
