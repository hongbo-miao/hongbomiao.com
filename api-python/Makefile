poetry-lock:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev
poetry-run-dev:
	poetry run poe dev
poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage
poetry-run-serve:
	poetry run poe serve

docker-build:
	cd .. && docker build --file=api-python/Dockerfile --tag=ghcr.io/hongbo-miao/hm-api-python:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-api-python:latest
docker-run:
	docker run --publish=35903:35903 --name=hm-api-python --rm ghcr.io/hongbo-miao/hm-api-python:latest
