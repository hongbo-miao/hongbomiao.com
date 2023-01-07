poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	brew install librdkafka
	export C_INCLUDE_PATH=/opt/homebrew/include
	export LIBRARY_PATH=/opt/homebrew/lib
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

poetry-run-dev:
	poetry run poe dev
poetry-run-serve:
	poetry run poe serve
poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage
poetry-run-flask-help:
	poetry run poe flask-help
poetry-run-flask-list-routes:
	poetry run poe flask-list-routes
poetry-run-flask-greet:
	poetry run poe flask-greet

docker-build:
	cd .. && \
	docker build --file=api-python/Dockerfile --tag=ghcr.io/hongbo-miao/hm-api-python:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-api-python:latest
docker-run:
	docker run --publish=35903:35903 --name=hm-api-python --rm ghcr.io/hongbo-miao/hm-api-python:latest
