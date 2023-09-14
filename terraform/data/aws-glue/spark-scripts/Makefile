poetry-env-use:
	poetry env use 3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

upload-spark-scripts:
	cd ../.. && \
	aws s3 cp aws-glue/spark-scripts/src/ s3://hongbomiao-bucket/aws-glue/spark-scripts/ --recursive
