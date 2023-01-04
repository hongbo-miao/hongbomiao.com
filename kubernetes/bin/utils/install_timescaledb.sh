#!/usr/bin/env bash
set -e

echo "# Install TimescaleDB"
# https://docs.timescale.com/install/latest/installation-kubernetes/#install-timescaledb-on-kubernetes
# https://github.com/FairwindsOps/charts/blob/master/stable/goldilocks/values.yaml
helm repo add timescale https://charts.timescale.com
helm repo update

kubectl apply --filename=kubernetes/manifests/timescale/hm-timescale-namespace.yaml
helm install \
  timescale \
  timescale/timescaledb-single \
  --namespace=hm-timescale \
  --values=kubernetes/manifests/timescale/helm/my-values.yaml

# Delete:
# helm uninstall timescale --namespace=hm-timescale
echo "=================================================="

kubectl port-forward service/timescale --namespace=hm-timescale 25495:25495 &
sleep 5

echo "# Create database in TimescaleDB"
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="create database twitter_db;"
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="grant all privileges on database twitter_db to admin;"
echo "=================================================="

# echo "# Migrate twitter_db in TimescaleDB"
# TIMESCALEDB_URL="postgresql://admin:passw0rd@localhost:25495/twitter_db?search_path=public"
# migrate -database "${TIMESCALEDB_URL}" -path kubernetes/data/timescaledb/twitter_db/migrations up
# echo "=================================================="

pgrep kubectl | xargs kill -9
