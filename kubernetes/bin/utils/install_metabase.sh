#!/usr/bin/env bash
set -e

echo "# Install Postgres"
source kubernetes/bin/utils/install_postgres.sh
psql postgresql://admin@localhost:5432/postgres --command="create database hm_metabase_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database hm_metabase_db to admin;"
echo "=================================================="

echo "# Install Metabase"
helm repo add pmint93 https://pmint93.github.io/helm-charts
helm repo update pmint93
helm install \
  metabase \
  pmint93/metabase \
  --namespace=hm-metabase \
  --create-namespace \
  --values=kubernetes/manifests/metabase/helm/my-values.yaml
# helm uninstall metabase --namespace=hm-metabase
echo "=================================================="
