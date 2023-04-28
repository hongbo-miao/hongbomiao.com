#!/usr/bin/env bash
set -e

echo "# Physical backup timescaledb"
sudo -u postgres pg_basebackup \
  --host=localhost \
  --port=5432 \
  --username=postgres \
  --pgdata="/var/lib/postgresql/backups/$(date +%F)/" \
  --wal-method=stream \
  --format=tar \
  --gzip \
  --progress
# Input Postgres password `passw0rd`
echo "=================================================="
