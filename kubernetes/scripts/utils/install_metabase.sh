#!/usr/bin/env bash
set -e

echo "# Install Postgres"
source kubernetes/bin/utils/install_postgres.sh
psql postgresql://admin@localhost:5432/postgres --command="create database hm_metabase_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database hm_metabase_db to admin;"
echo "=================================================="

echo "# Install Metabase"
helm upgrade \
  metabase \
  metabase \
  --install \
  --repo=https://pmint93.github.io/helm-charts \
  --namespace=hm-metabase \
  --create-namespace \
  --values=kubernetes/manifests/metabase/helm/my-values.yaml
# helm uninstall metabase --namespace=hm-metabase
# kubectl delete namespace hm-metabase
echo "=================================================="
