poetry-env-use:
	poetry env use python3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

upload-spark-scripts:
	cd ../.. && \
	aws s3 cp aws-glue/spark-scripts/src/ s3://hongbomiao-bucket/aws-glue/spark-scripts/ --recursive
