uv-install-python:
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --dev

uv-run-test:
	uv run poe test
uv-run-test-coverage:
	uv run poe test-coverage

upload-spark-scripts:
	cd ../.. && \
	aws s3 cp aws-glue/spark-scripts/src/ s3://hm-production-bucket/aws-glue/spark-scripts/ --recursive
