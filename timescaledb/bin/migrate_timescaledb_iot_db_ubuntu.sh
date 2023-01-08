#!/usr/bin/env bash
set -e

echo "# Create database iot_db"
psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="create database iot_db;"
psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="grant all privileges on database iot_db to postgres;"
echo "=================================================="

# Modify /etc/postgresql/14/main/postgresql.conf
#   shared_preload_libraries = 'timescaledb'
# Restart Postgres
#   sudo systemctl restart postgresql

echo "# Create extension timescaledb in iot_db"
psql postgresql://postgres:passw0rd@localhost:5432/iot_db --command="create extension if not exists timescaledb;"
echo "=================================================="

echo "# Migrate database iot_db"
TIMESCALEDB_URL="postgresql://postgres:passw0rd@localhost:5432/iot_db?sslmode=disable&search_path=public"
migrate -database "${TIMESCALEDB_URL}" -path timescaledb/migrations up
echo "=================================================="
