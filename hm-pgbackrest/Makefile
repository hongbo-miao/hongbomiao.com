docker-build:
	cd .. && \
	docker build --file=hm-pgbackrest/Dockerfile --tag=ghcr.io/hongbo-miao/hm-pgbackrest:latest .
docker-run:
	docker run --interactive --tty --rm --entrypoint=sh ghcr.io/hongbo-miao/hm-pgbackrest:latest

setup:
	sh bin/ubuntu/setup.sh

sql-show-pgbackrest-info:
	psql postgresql://postgres:passw0rd@localhost:5432/postgres --file=sql/pgsql-pgbackrest-info.sql

download-pgbackrest-sql:
	curl https://raw.githubusercontent.com/pgbackrest/pgbackrest/main/doc/example/pgsql-pgbackrest-info.sql > sql/pgsql-pgbackrest-info.sql
	curl https://raw.githubusercontent.com/pgbackrest/pgbackrest/main/doc/example/pgsql-pgbackrest-query.sql > sql/pgsql-pgbackrest-query.sql
