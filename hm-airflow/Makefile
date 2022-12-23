poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

helm-download-values:
	helm show values apache-airflow/airflow > values.yaml
helm-get-my-values:
	helm get values airflow --namespace=hm-airflow
