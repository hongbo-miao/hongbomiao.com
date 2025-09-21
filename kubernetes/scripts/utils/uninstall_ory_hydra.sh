#!/usr/bin/env bash
set -e

echo "# Uninstall ORY Hydra"
helm uninstall ory-hydra --namespace=hm-ory-hydra
kubectl delete --filename=kubernetes/manifests/ory-hydra/hm-ory-hydra-namespace.yaml
echo "=================================================="

echo "# Drop ory_hydra_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 5432:5432 &
sleep 3
psql postgresql://admin@localhost:5432/postgres --command="drop database if exists ory_hydra_db with (force);"
pgrep kubectl | xargs kill -9
echo "=================================================="
