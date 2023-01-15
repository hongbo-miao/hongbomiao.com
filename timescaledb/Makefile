# macOS
install-postgres:
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
	cd .. && \
	sh timescaledb/bin/macos/migrate_timescaledb_iot_db.sh

# Ubuntu
ubuntu-timescaledb-migrate:
	cd .. && \
	sh timescaledb/bin/ubuntu/migrate_timescaledb_iot_db.sh
ubuntu-timescaledb-logical-backup:
	cd .. && \
	sh timescaledb/bin/ubuntu/logical_backup_timescaledb_iot_db.sh
ubuntu-timescaledb-physical-backup:
	cd .. && \
	sh timescaledb/bin/ubuntu/physical_backup_timescaledb.sh
ubuntu-timescaledb-physical-restore:
	cd .. && \
	sh timescaledb/bin/ubuntu/physical_restore_timescaledb.sh

systemctl-list-dependencies:
	systemctl list-dependencies postgresql

sql-show-data-directory:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="show data_directory;"
sql-show-archive-mode:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="show archive_mode;"
sql-show-wal-level:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="show wal_level;"
