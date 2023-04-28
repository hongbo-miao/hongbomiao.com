#!/usr/bin/env bash
set -e

echo "# Create database hm_iot_db"
psql postgresql://admin:passw0rd@localhost:16863/postgres --command="create database hm_iot_db;"
psql postgresql://admin:passw0rd@localhost:16863/postgres --command="grant all privileges on database hm_iot_db to admin;"
echo "=================================================="

echo "# Migrate database hm_iot_db"
TIMESCALEDB_URL="postgresql://admin:passw0rd@localhost:16863/hm_iot_db?sslmode=disable&search_path=public"
migrate -database "${TIMESCALEDB_URL}" -path migrations up
echo "=================================================="
