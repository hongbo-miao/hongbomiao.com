#!/usr/bin/env bash
set -e

echo "# Uninstall Superset"
helm uninstall superset --namespace=hm-superset
kubectl delete jobs superset-init-db --namespace=hm-superset
kubectl delete --filename=kubernetes/manifests/superset/hm-superset-namespace.yaml
echo "=================================================="

echo "# Drop superset_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 5432:5432 &
sleep 3
psql postgresql://admin@localhost:5432/postgres --command="drop database if exists superset_db with (force);"
pgrep kubectl | xargs kill -9
echo "=================================================="
