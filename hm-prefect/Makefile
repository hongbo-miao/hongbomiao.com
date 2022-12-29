poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

# Prefect
prefect-local:
	poetry run poe prefect-local
prefect-cloud-login:
	poetry run poe prefect-cloud-login -- --key=xxx
prefect-prefect-cloud-workspace-set:
	poetry run poe prefect-cloud-workspace-set
prefect-agent-start:
	poetry run poe prefect-agent-start

dev-calculate:
	poetry run poe dev-calculate
dev-find-taxi-top-routes:
	poetry run poe dev-find-taxi-top-routes
dev-greet:
	poetry run poe dev-greet
dev-collect-data:
	poetry run poe dev-collect-data
build-collect-data:
	poetry run poe build-collect-data -- --params=$$(cat src/collect_data/params.json | jq -c .)
deploy-collect-data:
	poetry run poe deploy-collect-data
run-collect-data:
	poetry run poe run-collect-data

test:
	poetry run poe test
test-coverage:
	poetry run poe test-coverage
