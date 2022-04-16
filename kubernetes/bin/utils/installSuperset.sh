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
kubectl apply --filename=kubernetes/config/superset/hm-superset-namespace.yaml
helm repo add superset https://apache.github.io/superset
helm install \
  superset \
  --namespace=hm-superset \
  --values=kubernetes/config/superset/helm/values.yaml \
  --values=kubernetes/config/superset/helm/my-values.yaml \
  superset/superset


# Upgrade:
# helm upgrade superset \
#   --namespace=hm-superset \
#   --values=kubernetes/config/superset/helm/values.yaml \
#   --values=kubernetes/config/superset/helm/my-values.yaml \
#   superset/superset

# Delete:
# helm uninstall --namespace=hm-superset superset
echo "=================================================="
