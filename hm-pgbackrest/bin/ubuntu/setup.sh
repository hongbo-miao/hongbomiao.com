#!/usr/bin/env bash
set -e

# https://pgbackrest.org/user-guide.html

echo "# Build pgBackRest in the host (macOS)"
# docker create --name=hm-pgbackrest ghcr.io/hongbo-miao/hm-pgbackrest:latest
# docker cp hm-pgbackrest:/usr/src/app/build/pgbackrest-release-2.43/src/pgbackrest pgbackrest
# rclone copy --progress pgbackrest hm-ubuntu:/tmp/
echo "=================================================="

echo "# Install pgBackRest"
sudo cp /tmp/pgbackrest /usr/bin/pgbackrest
rm -f /tmp/pgbackrest
sudo chmod 755 /usr/bin/pgbackrest

sudo mkdir --parents --mode=770 /var/log/pgbackrest
sudo chown postgres:postgres /var/log/pgbackrest
sudo mkdir --parents /etc/pgbackrest
sudo mkdir --parents /etc/pgbackrest/conf.d
sudo touch /etc/pgbackrest/pgbackrest.conf
sudo chmod 640 /etc/pgbackrest/pgbackrest.conf
sudo chown postgres:postgres /etc/pgbackrest/pgbackrest.conf
sudo chown parallels:parallels /tmp/pgbackrest.conf
# Check: sudo -u postgres pgbackrest
echo "=================================================="

echo "# Install TimescaleDB"
# Follow "Install TimescaleDB" in timescaledb/bin/ubuntu/install_timescaledb.sh
echo "=================================================="

echo "# Setup demo cluster"
sudo -u postgres /usr/lib/postgresql/14/bin/initdb --pgdata=/var/lib/postgresql/14/demo --data-checksums --auth=peer
sudo pg_createcluster 14 demo
echo "=================================================="

echo "# Create the repository"
sudo mkdir --parents /var/lib/pgbackrest
sudo chmod 750 /var/lib/pgbackrest
sudo chown postgres:postgres /var/lib/pgbackrest
echo "=================================================="

echo "# Configure postgresql.conf"
# Check: sudo nano /etc/postgresql/14/demo/postgresql.conf
# shared_preload_libraries = 'timescaledb'
sudo perl -pi -e "s/#shared_preload_libraries = ''/shared_preload_libraries = 'timescaledb'/" /etc/postgresql/14/demo/postgresql.conf
# archive_mode = on
sudo perl -pi -e "s/#archive_mode = off/archive_mode = on/" /etc/postgresql/14/demo/postgresql.conf
# wal_level = replica
sudo perl -pi -e "s/#wal_level = replica/wal_level = replica/" /etc/postgresql/14/demo/postgresql.conf
# max_wal_senders = 3
sudo perl -pi -e "s/#max_wal_senders = 10/max_wal_senders = 3/" /etc/postgresql/14/demo/postgresql.conf
# archive_command = 'pgbackrest --stanza=demo archive-push %p'
sudo perl -pi -e "s/#archive_command = ''/archive_command = 'pgbackrest --stanza=demo archive-push %p'/" /etc/postgresql/14/demo/postgresql.conf

sudo pg_ctlcluster 14 demo restart
sudo systemctl restart postgresql@14-demo
echo "=================================================="

echo "# Configure pgbackrest.conf"
printf "[demo]\npg1-path=/var/lib/postgresql/14/demo\n" | sudo -u postgres tee --append /etc/pgbackrest/pgbackrest.conf
printf "[global:archive-push]\ncompress-level=3\n" | sudo -u postgres tee --append /etc/pgbackrest/pgbackrest.conf
printf "[global]\nrepo1-cipher-pass=0ZaLcF26HG6Z7V2hDQjMgBhY8joYpaqWD607WgLgaY0X6bnXGMzfOYJCzoeIB6X4\nrepo1-cipher-type=aes-256-cbc\nrepo1-path=/var/lib/pgbackrest\nrepo1-retention-full=2\nstart-fast=y\n" | sudo -u postgres tee --append /etc/pgbackrest/pgbackrest.conf
echo "=================================================="

echo "# Create the Stanza"
sudo -u postgres pgbackrest --stanza=demo --log-level-console=info stanza-create
echo "=================================================="

echo "# Check the configuration"
sudo -u postgres pgbackrest --stanza=demo --log-level-console=info check
echo "=================================================="

echo "# Perform a backup"
sudo perl -pi -e "s/#max_wal_senders = 10/max_wal_senders = 3/" /etc/postgresql/14/demo/postgresql.conf
# Full backup
sudo -u postgres pgbackrest --stanza=demo --type=full --log-level-console=info backup
# Incremental backup (Default. If no full backup existed pgBackRest will run a full backup instead)
sudo -u postgres pgbackrest --stanza=demo --log-level-console=info backup
# Differential backup
sudo -u postgres pgbackrest --stanza=demo --type=diff --log-level-console=info backup
echo "=================================================="

echo "# Check backups"
sudo -u postgres pgbackrest info
echo "=================================================="
