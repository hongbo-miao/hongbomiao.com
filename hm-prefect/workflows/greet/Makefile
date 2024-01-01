poetry-env-use:
	poetry env use python3.11
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-prefect-server-start:
	poetry run poe prefect-server-start
poetry-run-prefect-agent-start:
	poetry run poe prefect-agent-start
prefect-profile-use:
	poetry run poe prefect-profile-use -- hm-prefect

poetry-run-set-up:
	poetry run poe set-up
poetry-run-dev:
	poetry run poe dev
poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage
poetry-run-build:
	poetry run poe build -- --params='{"user":{"first_name":"Hongbo","last_name":"Miao"}}' --work-queue=hm-kubernetes-queue
poetry-run-run:
	poetry run poe run

docker-build:
	cd ../../.. && \
	docker build --file=hm-prefect/workflows/greet/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-greet:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-prefect-greet:latest
