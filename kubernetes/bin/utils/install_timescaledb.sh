#!/usr/bin/env bash
set -e

echo "# Install TimescaleDB"
# https://docs.timescale.com/install/latest/installation-kubernetes/#install-timescaledb-on-kubernetes
helm upgrade \
  timescale \
  timescaledb-single \
  --install \
  --repo=https://charts.timescale.com \
  --namespace=hm-timescale \
  --create-namespace \
  --values=kubernetes/manifests/timescaledb/helm/my-values.yaml
# helm uninstall timescale --namespace=hm-timescale
# kubectl delete namespace hm-timescale
echo "=================================================="

echo "# Create database twitter_db"
kubectl port-forward service/timescale --namespace=hm-timescale 25495:5432 &
sleep 5
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="create database twitter_db;"
psql postgresql://admin:passw0rd@localhost:25495/postgres --command="grant all privileges on database twitter_db to admin;"
echo "=================================================="

# echo "# Migrate database twitter_db"
# TIMESCALEDB_URL="postgresql://admin:passw0rd@localhost:25495/twitter_db?search_path=public"
# migrate -database "${TIMESCALEDB_URL}" -path streaming/migrations up
# echo "=================================================="

pgrep kubectl | xargs kill -9
