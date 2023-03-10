#!/usr/bin/env bash
set -e

echo "# Install Postgres"
kubectl apply --filename=kubernetes/manifests/postgres
echo "=================================================="

kubectl wait pod postgres-statefulset-0 --for=condition=ready --namespace=hm-postgres

kubectl port-forward service/postgres-service --namespace=hm-postgres 5432:5432 &
sleep 5

echo "# Create database in Postgres"
psql postgresql://admin@localhost:5432/postgres --command="create database opa_db;"
psql postgresql://admin@localhost:5432/postgres --command="create database ory_hydra_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database opa_db to admin;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database ory_hydra_db to admin;"
echo "=================================================="

# echo "# Migrate opa_db in Postgres"
# POSTGRESQL_URL="postgresql://admin:passw0rd@localhost:5432/opa_db?sslmode=disable&search_path=public"
# migrate -database "${POSTGRESQL_URL}" -path kubernetes/data/postgres/opa_db/migrations up
# echo "=================================================="

pgrep kubectl | xargs kill -9
