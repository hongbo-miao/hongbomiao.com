poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

poetry-run-prefect-server-start:
	poetry run poe prefect-server-start
poetry-run-prefect-agent-start:
	poetry run poe prefect-agent-start
prefect-profile-use:
	poetry run poe prefect-profile-use -- hm-kubernetes

poetry-run-concurrency-limit-create:
	poetry run poe prefect-concurrency-limit-create -- upload_to_influxdb_concurrency_limit 2
poetry-run-dev:
	poetry run poe dev
poetry-run-add-kubernetes-job-block:
	poetry run poe add-kubernetes-job-block
poetry-run-build:
	poetry run poe build -- --params=$$(cat params.json | jq -c .) --work-queue=hm-kubernetes-queue
poetry-run-run:
	poetry run poe run

docker-build:
	cd ../../.. && \
	docker build --file=hm-prefect/workflows/upload-to-influxdb/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-upload-to-influxdb:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-prefect-upload-to-influxdb:latest
