#!/usr/bin/env bash
set -e

# https://docs.timescale.com/install/latest/self-hosted/installation-debian/

# Install TimescaleDB
sudo apt install gnupg postgresql-common apt-transport-https lsb-release wget
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh
echo "deb https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -c -s) main" | sudo tee /etc/apt/sources.list.d/timescaledb.list
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -

# Install psql
sudo apt update
sudo apt install timescaledb-2-postgresql-14

# Set up the TimescaleDB extension
sudo systemctl restart postgresql
sudo -u postgres psql
# \password postgres
# \q
