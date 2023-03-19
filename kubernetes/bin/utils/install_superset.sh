#!/usr/bin/env bash
set -e

echo "# Create hm_superset_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 5432:5432 &
sleep 5
psql postgresql://admin@localhost:5432/postgres --command="create database hm_superset_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database hm_superset_db to admin;"
pgrep kubectl | xargs kill -9
echo "=================================================="

echo "# Install Superset"
helm repo add superset https://apache.github.io/superset
helm repo update superset
helm install \
  superset \
  superset/superset \
  --namespace=hm-superset \
  --create-namespace \
  --values=kubernetes/manifests/superset/helm/my-values.yaml
# helm uninstall superset --namespace=hm-superset

# Install database drivers:
# https://superset.apache.org/docs/databases/installing-database-drivers
echo "=================================================="
