poetry-env-use:
	poetry env use 3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

helm-download-values:
	helm show values apache-airflow/airflow > values.yaml
helm-get-my-values:
	helm get values airflow --namespace=hm-airflow
