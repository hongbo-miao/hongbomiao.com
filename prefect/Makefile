conda-create:
	conda create --name=hm-prefect python=3.10
conda-env-remove:
	conda env remove --name=hm-prefect
conda-activate:
	conda activate hm-prefect
conda-deactivate:
	conda deactivate
conda-env-list:
	conda env list
conda-list-packages:
	conda list

pip-install:
	pip install --requirement=requirements.txt

prefect-local:
	prefect orion start
prefect-cloud:
	prefect cloud login --key=xxx
	prefect cloud workspace set --workspace=hongbo-miao/hongbomiao

workflow-start:
	python greet.py
	python taxi/get_taxi_statistics.py
