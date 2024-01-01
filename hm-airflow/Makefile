poetry-env-use:
	poetry env use python3.11
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

helm-download-values:
	helm show values apache-airflow/airflow > values.yaml
helm-get-my-values:
	helm get values airflow --namespace=hm-airflow
