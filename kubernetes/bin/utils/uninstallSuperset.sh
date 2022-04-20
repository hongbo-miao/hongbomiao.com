#!/usr/bin/env bash
set -e

echo "# Uninstall Superset"
helm uninstall --namespace=hm-superset superset
kubectl delete jobs superset-init-db --namespace=hm-superset
kubectl delete --filename=kubernetes/manifests/superset/hm-superset-namespace.yaml
echo "=================================================="

echo "# Drop superset_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072 &
sleep 3
psql --host=localhost --port=40072 --dbname=postgres --username=admin --command="drop database if exists superset_db with (force);"
pgrep kubectl | xargs kill -9
echo "=================================================="
