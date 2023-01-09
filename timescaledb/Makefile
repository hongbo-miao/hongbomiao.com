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
	sh timescaledb/bin/migrate_timescaledb_iot_db.sh

# Ubuntu
timescaledb-migrate-ubuntu:
	cd .. && \
	sh timescaledb/bin/migrate_timescaledb_iot_db_ubuntu.sh
timescaledb-logical-backup:
	cd .. && \
	sh timescaledb/bin/logical_backup_timescaledb_iot_db.sh

sql-show-data-directory:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="show data_directory;"
