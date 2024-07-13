poetry-env-use:
	poetry env use python3.9
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-dev:
	poetry run poe dev
poetry-run-submit:
	# https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html
	poetry run poe submit -- --address=https://hm-ray.internal.hongbomiao.com --working-dir=src -- python main.py
