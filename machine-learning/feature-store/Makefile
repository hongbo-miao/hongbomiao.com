poetry-env-use:
	poetry env use python3.11
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

feast-init:
	feast init driver_features
download-driver-stats:
	curl --silent --fail --show-error --location https://raw.githubusercontent.com/hongbo-miao/hm-driver-stats/main/driver_stats.parquet > features/data/driver_stats.parquet

poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

poetry-run-inspect-raw-data:
	poetry run poe inspect-raw-data
# Scan python files in the current directory for feature view/entity definitions, registers the objects, and deploys infrastructure.
poetry-run-feast-apply:
	poetry run poe feast-apply
# Generate training data
poetry-run-generate-training-data:
	poetry run poe generate-training-data
# Load features into your online store
poetry-run-materialize-features:
	poetry run poe materialize-features
# Inspect materialized features
poetry-run-inspect-features:
	poetry run poe inspect-features
# Fetch feature vectors for inference
poetry-run-fetch-features:
	poetry run poe fetch-features
# Fetch feature vectors for inference by feature service
poetry-run-fetch-features-by-service:
	poetry run poe fetch-features-by-service
poetry-run-feast-ui:
	poetry run poe feast-ui
