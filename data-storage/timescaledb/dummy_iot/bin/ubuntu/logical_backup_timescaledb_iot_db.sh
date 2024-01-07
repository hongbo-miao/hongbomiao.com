#!/usr/bin/env bash
set -e

echo "# Logical backup database iot_db"
sudo pg_dump \
  --host=localhost \
  --port=5432 \
  --username=postgres \
  --format=c \
  --file=iot_db.bak \
  iot_db
echo "=================================================="
