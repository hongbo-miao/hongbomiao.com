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
	/opt/homebrew/Cellar/postgresql@14/14.6/bin/pg_dump \
		--host=localhost \
		--port=5432 \
		--username=postgres \
		--format=c \
		--file=iot_db.bak \
		iot_db
timescaledb-physical-backup:
	/opt/homebrew/Cellar/postgresql@14/14.6/bin/pg_basebackup \
		--host=localhost \
		--port=5432 \
		--username=postgres \
		--pgdata=/tmp/timescaledb-backup \
		--wal-method=stream \
		--format=tar \
		--gzip \
		--progress
