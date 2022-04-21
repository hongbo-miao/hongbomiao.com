#!/usr/bin/env bash
set -e

echo "# Create superset_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072 &
sleep 5
psql --host=localhost --port=40072 --dbname=postgres --username=admin --command="create database superset_db;"
psql --host=localhost --port=40072 --dbname=postgres --username=admin --command="grant all privileges on database superset_db to admin;"
pgrep kubectl | xargs kill -9
echo "=================================================="

echo "# Install Superset"
kubectl apply --filename=kubernetes/manifests/superset/hm-superset-namespace.yaml
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
# helm uninstall --namespace=hm-superset superset
echo "=================================================="
