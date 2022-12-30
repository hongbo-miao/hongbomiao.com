poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

poetry-run-prefect-orion-start:
	poetry run poe prefect-orion-start
poetry-run-prefect-agent-start:
	poetry run poe prefect-agent-start

poetry-run-dev-calculate:
	poetry run poe dev-calculate
poetry-run-dev-find-taxi-top-routes:
	poetry run poe dev-find-taxi-top-routes
poetry-run-dev-greet:
	poetry run poe dev-greet

poetry-run-dev-collect-data:
	poetry run poe dev-collect-data
poetry-run-build-collect-data:
	poetry run poe build-collect-data -- --params=$$(cat src/collect_data/params.json | jq -c .)
poetry-run-run-collect-data:
	poetry run poe run-collect-data

poetry-run-dev-print-platform:
	poetry run poe dev-print-platform
poetry-run-build-print-platform:
	poetry run poe build-print-platform
poetry-run-run-print-platform:
	poetry run poe run-print-platform

poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

# Prefect
prefect-profile-ls:
	prefect profile ls
prefect-profile-use:
	prefect profile use default
prefect-profile-create:
	prefect profile create xxx
prefect-profile-delete:
	prefect profile delete xxx

prefect-queue-ls:
	prefect work-queue ls
prefect-queue-create:
	prefect work-queue create hm-queue
prefect-queue-delete:
	prefect work-queue delete hm-queue

prefect-config-view:
	prefect config view
prefect-config-view-show-defaults:
	prefect config view --show-defaults
prefect-config-set:
	prefect config set PREFECT_API_URL=http://localhost:4200/api

# Prefect Cloud
prefect-cloud-login:
	prefect cloud login
prefect-cloud-workspace-set:
	prefect cloud workspace set --workspace=hongbo-miao/hongbomiao
