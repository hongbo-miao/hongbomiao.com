#!/usr/bin/env bash
set -e

# https://docs.timescale.com/install/latest/self-hosted/installation-debian/

echo "# Install TimescaleDB"
sudo apt install --yes gnupg postgresql-common apt-transport-https lsb-release wget
sudo YES=yes /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
echo "=================================================="

echo "# Install psql"
sudo apt update
sudo apt install --yes timescaledb-2-postgresql-14
sudo systemctl restart postgresql
sudo -u postgres psql
# Set password to `passw0rd`: \password postgres
# Quit: \q
echo "=================================================="

echo "# Set up postgresql.conf"
# Check: sudo nano /etc/postgresql/14/main/postgresql.conf

# shared_preload_libraries = 'timescaledb'
sudo perl -pi -e "s/#shared_preload_libraries = ''/shared_preload_libraries = 'timescaledb'/" /etc/postgresql/14/main/postgresql.conf
# archive_mode = on
sudo perl -pi -e "s/#archive_mode = off/archive_mode = on/" /etc/postgresql/14/main/postgresql.conf
# wal_level = replica
sudo perl -pi -e "s/#wal_level = replica/wal_level = replica/" /etc/postgresql/14/main/postgresql.conf
# archive_command = 'cp %p /var/lib/postgresql/pg_wal_archive/%f'
sudo perl -pi -e "s/#archive_command = ''/archive_command = 'cp %p \/var\/lib\/postgresql\/pg_wal_archive\/%f'/" /etc/postgresql/14/main/postgresql.conf

sudo -u postgres mkdir /var/lib/postgresql/pg_wal_archive/
sudo systemctl restart postgresql
echo "=================================================="
