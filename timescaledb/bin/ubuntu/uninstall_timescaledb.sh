#!/usr/bin/env bash
set -e

echo "# Uninstall TimescaleDB"
sudo systemctl stop postgresql
sudo apt purge --yes postgresql postgresql-*
sudo rm -r /var/lib/postgresql/backups/
sudo rm -r /var/lib/postgresql/pg_wal_archive/
echo "=================================================="
