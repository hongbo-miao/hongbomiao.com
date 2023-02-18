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

# calculate
poetry-run-dev-calculate:
	poetry run poe dev-calculate

# collect-data
poetry-run-dev-collect-data:
	poetry run poe dev-collect-data
poetry-run-build-collect-data:
	poetry run poe build-collect-data -- --params=$$(cat src/collect_data/params.json | jq -c .)
poetry-run-run-collect-data:
	poetry run poe run-collect-data

# find-taxi-top-routes
poetry-run-dev-find-taxi-top-routes:
	poetry run poe dev-find-taxi-top-routes

# greet
poetry-run-dev-greet:
	poetry run poe dev-greet

# print-platform
poetry-run-dev-print-platform:
	poetry run poe dev-print-platform
poetry-run-build-print-platform:
	poetry run poe build-print-platform

docker-build-hm-prefect-print-platform:
	cd .. && \
	docker build --file=hm-prefect/build/print_platform/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-print-platform:latest .
docker-push-hm-prefect-print-platform:
	docker push ghcr.io/hongbo-miao/hm-prefect-print-platform:latest
poetry-run-add-kubernetes-job-block-print-platform:
	poetry run poe add-kubernetes-job-block-print-platform
poetry-run-build-kubernetes-print-platform:
	poetry run poe build-kubernetes-print-platform

poetry-run-run-print-platform:
	poetry run poe run-print-platform

# Test
poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

# Prefect
prefect-profile-ls:
	prefect profile ls
prefect-profile-use:
	prefect profile use default
	prefect profile use xxx
prefect-profile-create:
	prefect profile create xxx
prefect-profile-delete:
	prefect profile delete xxx
prefect-profile-inspect:
	prefect profile inspect xxx

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
