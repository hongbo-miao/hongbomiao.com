#!/usr/bin/env bash
set -e

kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072 &
sleep 5

echo "# Create database in Postgres"
psql --host=localhost --port=40072 --dbname=postgres --username=admin --command="create database opa_db;"
psql --host=localhost --port=40072 --dbname=postgres --username=admin --command="grant all privileges on database opa_db to admin;"
echo "=================================================="

echo "# Initialize opa_db in Postgres"
POSTGRESQL_URL="postgres://admin:passw0rd@localhost:40072/opa_db?sslmode=disable&search_path=public"
migrate -database "${POSTGRESQL_URL}" -path kubernetes/data/postgres/opa_db/migrations up
echo "=================================================="

pgrep kubectl | xargs kill -9
