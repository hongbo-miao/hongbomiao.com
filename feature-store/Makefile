poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

feast-init:
	feast init driver_features
download-driver-stats:
	curl https://raw.githubusercontent.com/Hongbo-Miao/hm-driver-stats/main/driver_stats.parquet > features/data/driver_stats.parquet

inspect-raw-data:
	poetry run poe inspect-raw-data
# Scan python files in the current directory for feature view/entity definitions, registers the objects, and deploys infrastructure.
feast-apply:
	poetry run poe feast-apply
# Generate training data
generate-training-data:
	poetry run poe generate-training-data
# Load features into your online store
materialize-features:
	poetry run poe materialize-features
# Inspect materialized features
inspect-features:
	poetry run poe inspect-features
# Fetch feature vectors for inference
fetch-features:
	poetry run poe fetch-features
# Fetch feature vectors for inference by feature service
fetch-features-by-service:
	poetry run poe fetch-features-by-service
feast-ui:
	poetry run poe feast-ui
