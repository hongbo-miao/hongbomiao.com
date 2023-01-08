#!/usr/bin/env bash
set -e

echo "# Create database in TimescaleDB"
psql postgresql://admin:passw0rd@localhost:16863/postgres --command="create database iot_db;"
psql postgresql://admin:passw0rd@localhost:16863/postgres --command="grant all privileges on database iot_db to admin;"
echo "=================================================="

echo "# Migrate iot_db in TimescaleDB"
TIMESCALEDB_URL="postgresql://admin:passw0rd@localhost:16863/iot_db?sslmode=disable&search_path=public"
migrate -database "${TIMESCALEDB_URL}" -path timescaledb/migrations up
echo "=================================================="
