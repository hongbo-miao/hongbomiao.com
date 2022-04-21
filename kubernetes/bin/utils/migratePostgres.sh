#!/usr/bin/env bash
set -e

kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072 &
sleep 5

echo "# Create database in Postgres"
psql postgresql://admin@localhost:40072/postgres --command="create database opa_db;"
psql postgresql://admin@localhost:40072/postgres --command="create database ory_hydra_db;"
psql postgresql://admin@localhost:40072/postgres --command="grant all privileges on database opa_db to admin;"
psql postgresql://admin@localhost:40072/postgres --command="grant all privileges on database ory_hydra_db to admin;"
echo "=================================================="

echo "# Migrate opa_db in Postgres"
POSTGRESQL_URL="postgres://admin:passw0rd@localhost:40072/opa_db?sslmode=disable&search_path=public"
migrate -database "${POSTGRESQL_URL}" -path kubernetes/data/postgres/opa_db/migrations up
echo "=================================================="

pgrep kubectl | xargs kill -9
