poetry-env-use:
	poetry env use python3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

upload-spark-scripts:
	cd ../.. && \
	aws s3 cp aws-glue/spark-scripts/src/ s3://hm-production-bucket/aws-glue/spark-scripts/ --recursive
