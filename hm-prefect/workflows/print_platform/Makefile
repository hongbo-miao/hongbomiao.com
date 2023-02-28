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

poetry-run-dev:
	poetry run poe dev
poetry-run-add-kubernetes-job-block:
	poetry run poe add-kubernetes-job-block
poetry-run-build:
	poetry run poe build -- --work-queue=hm-local-queue
poetry-run-run:
	poetry run poe run

docker-build:
	cd ../../.. && \
	docker build --file=hm-prefect/workflows/print_platform/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-print-platform:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-prefect-print-platform:latest
