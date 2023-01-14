#!/usr/bin/env bash
set -e

# https://docs.timescale.com/install/latest/self-hosted/installation-debian/

echo "# Install TimescaleDB"
sudo apt install --yes gnupg postgresql-common apt-transport-https lsb-release wget
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
echo "deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
echo "=================================================="

echo "# Install psql"
sudo apt update
sudo apt install --yes timescaledb-2-postgresql-14
sudo systemctl restart postgresql
sudo -u postgres psql
# \password postgres
# \q
echo "=================================================="

echo "# Set up the TimescaleDB extension"
sudo nano /etc/postgresql/14/main/postgresql.conf
# shared_preload_libraries = 'timescaledb'
sudo systemctl restart postgresql
echo "=================================================="
