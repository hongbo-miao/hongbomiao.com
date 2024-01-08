#!/usr/bin/env bash
set -e

echo "# Create database twitter_db"
psql postgresql://admin:passw0rd@localhost:16863/postgres --command="create database twitter_db;"
psql postgresql://admin:passw0rd@localhost:16863/postgres --command="grant all privileges on database twitter_db to admin;"
echo "=================================================="

echo "# Migrate database twitter_db"
TIMESCALEDB_URL="postgresql://admin:passw0rd@localhost:16863/twitter_db?sslmode=disable&search_path=public"
migrate -database "${TIMESCALEDB_URL}" -path data-processing/flink/applications/stream-tweets/migrations up
echo "=================================================="
