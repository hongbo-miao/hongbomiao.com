#!/usr/bin/env bash
set -e

echo "# Create superset_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072 &
sleep 5
psql postgresql://admin@localhost:40072/postgres --command="create database superset_db;"
psql postgresql://admin@localhost:40072/postgres --command="grant all privileges on database superset_db to admin;"
pgrep kubectl | xargs kill -9
echo "=================================================="

echo "# Install Superset"
kubectl apply --filename=manifests/superset/hm-superset-namespace.yaml
helm repo add superset https://apache.github.io/superset
helm install \
  superset \
  superset/superset \
  --namespace=hm-superset \
  --values=kubernetes/manifests/superset/helm/my-values.yaml

# Upgrade:
# helm upgrade \
#   superset \
#   superset/superset \
#   --namespace=hm-superset \
#   --values=kubernetes/manifests/superset/helm/my-values.yaml

# Delete:
# helm uninstall superset --namespace=hm-superset

# Install database drivers:
# https://superset.apache.org/docs/databases/installing-database-drivers
echo "=================================================="
