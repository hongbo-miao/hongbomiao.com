#!/usr/bin/env bash
set -e

echo "# Physical restore timescaledb"
sudo -i -u postgres

# Folders
# /etc/postgresql/14/main/pg_wal/
# /var/lib/postgresql/pg_wal_archive/
# /var/lib/postgresql/backups/

# psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="select pg_switch_wal();"

systemctl stop postgresql
cp --recursive /var/lib/postgresql/14/main/ /var/lib/postgresql/backups/main/
# Partial: cp --recursive /var/lib/postgresql/14/main/pg_wal/ /var/lib/postgresql/backups/pg_wal/

rm -r /var/lib/postgresql/14/main/*
tar -x -f /var/lib/postgresql/backups/2023-01-14/base.tar.gz -C /var/lib/postgresql/14/main/
rm -r /var/lib/postgresql/14/main/pg_wal/
cp --recursive /var/lib/postgresql/backups/main/pg_wal/ /var/lib/postgresql/14/main/pg_wal/
# Partial: sudo cp --recursive /var/lib/postgresql/backups/pg_wal/ /var/lib/postgresql/14/main/pg_wal/

# Check: nano /etc/postgresql/14/main/postgresql.conf
# restore_command = 'cp /var/lib/postgresql/pg_wal_archive/%f %p'
perl -pi -e "s/#restore_command = ''/restore_command = 'cp \/var\/lib\/postgresql\/pg_wal_archive\/%f %p'/" /etc/postgresql/14/main/postgresql.conf
# recovery_target_time = '2023-01-14 16:35:00'
perl -pi -e "s/#recovery_target_time = ''/recovery_target_time = '2023-01-14 20:23:00'/" /etc/postgresql/14/main/postgresql.conf

touch /var/lib/postgresql/14/main/recovery.signal
systemctl start postgresql

# Logs
# systemctl status postgresql@14-main.service
# journalctl -xeu postgresql@14-main.service
# cat /var/log/postgresql/postgresql-14-main.log
echo "=================================================="
