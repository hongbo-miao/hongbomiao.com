poetry-env-use:
	poetry env use python3.9
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-train:
	poetry run poe train
poetry-run-test:
	poetry run poe test
poetry-run-test-story:
	poetry run poe test -- --stories tests/test_stories.yml
poetry-run-visualize:
	poetry run poe visualize

poetry-run-start-rasa-server:
	poetry run poe start-rasa-server
poetry-run-start-action-server:
	poetry run poe start-action-server

poetry-run-telemetry-disable:
	poetry run poe telemetry-disable
poetry-run-shell:
	poetry run poe shell
poetry-run-interactive:
	poetry run poe interactive

# Docker
docker-build:
	cd ../.. && \
	docker build --file=machine-learning/hm-rasa/Dockerfile --tag=ghcr.io/hongbo-miao/hm-rasa-action-server:latest .
docker-push:
	docker push hongbomiao/hm-rasa-action-server:latest
