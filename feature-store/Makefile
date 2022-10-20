conda-create:
	conda create --name=hm-feature-store python=3.10 --yes
conda-env-remove:
	conda env remove --name=hm-feature-store
conda-activate:
	conda activate hm-feature-store
conda-deactivate:
	conda deactivate
conda-env-list:
	conda env list
conda-list-packages:
	conda list

pip-install:
	pip install --requirement=requirements.txt

feast-init:
	feast init driver_features

download-driver-stats:
	curl https://raw.githubusercontent.com/Hongbo-Miao/hm-driver-stats/main/driver_stats.parquet > features/data/driver_stats.parquet

python-inspect-raw-data:
	python inspect_raw_data.py

# Scan python files in the current directory for feature view/entity definitions, registers the objects, and deploys infrastructure.
feast-apply:
	cd driver_features && \
	feast apply

# Generate training data
python-generate:
	python generate_training_data.py

# Load features into your online store
feast-materialize-features:
	cd driver_features && \
	feast materialize-incremental $$(date -u +"%Y-%m-%dT%H:%M:%S")

# Inspect materialized features
python-inspect-features:
	python inspect_materialized_features.py

# Fetch feature vectors for inference
python-fetch-features:
	python fetch_features.py
# Fetch feature vectors for inference by feature service
python-fetch-features-by-service:
	python fetch_features_by_service.py

feast-ui:
	cd driver_features && \
	feast ui
