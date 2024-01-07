# macOS
postgres-install:
	brew install postgresql@14
	brew install postgresql@15
timescaledb-start:
	docker run \
		--detach \
		--name=timescaledb \
		--publish=16863:5432 \
		--env=POSTGRES_DB=postgres \
		--env=POSTGRES_USER=admin \
		--env=POSTGRES_PASSWORD=passw0rd \
		timescale/timescaledb:latest-pg15
timescaledb-migrate:
	sh bin/macos/migrate_timescaledb_iot_db.sh

# Ubuntu
ubuntu-timescaledb-migrate:
	sh bin/ubuntu/migrate_timescaledb_iot_db.sh
ubuntu-timescaledb-logical-backup:
	sh bin/ubuntu/logical_backup_timescaledb_iot_db.sh
ubuntu-timescaledb-physical-backup:
	sh bin/ubuntu/physical_backup_timescaledb.sh
ubuntu-timescaledb-physical-restore:
	sh bin/ubuntu/physical_restore_timescaledb.sh

psql:
	sudo -u postgres psql
    # Set password: \password postgres
    # List users and roles: \du
    # List databases: \l
    # List schemas: \dt
    # List functions: \df
    # List views: \dv
    # Quit: \q

pg-list-dependencies:
	systemctl list-dependencies postgresql
pg-list-cluster:
	pg_lsclusters
pg-stop-cluster:
	# Stopping the cluster using pg_ctlcluster will mark the systemd unit as failed. Using systemctl instead.
	# Avoid: sudo pg_ctlcluster 14 main stop
	sudo systemctl stop postgresql@14-main
pg-drop-cluster:
	sudo pg_dropcluster --stop 14 main
pg-status-cluster:
	sudo systemctl status postgresql@14-main

sql-show-data-directory:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="show data_directory;"
sql-show-archive-mode:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="show archive_mode;"
sql-show-wal-level:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="show wal_level;"
