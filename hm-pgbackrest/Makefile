docker-build:
	cd .. && \
	docker build --file=hm-pgbackrest/Dockerfile --tag=ghcr.io/hongbo-miao/hm-pgbackrest:latest .
docker-run:
	docker run --interactive --tty --rm --entrypoint=sh ghcr.io/hongbo-miao/hm-pgbackrest:latest

setup:
	sh bin/ubuntu/setup.sh
