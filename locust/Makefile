conda-create:
	conda create --name=hm-locust python=3.9
conda-env-remove:
	conda env remove --name=hm-locust
conda-activate:
	conda activate hm-locust
conda-deactivate:
	conda deactivate

poetry-install:
	poetry install
poetry-run-locust:
	poetry run poe locust
poetry-lock:
	poetry lock --no-update
