poetry-env-use:
	poetry env use 3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-dev:
	poetry run poe dev


zip-pytorch-mnist:
	cd .. && \
	zip -r pytorch-mnist.zip pytorch-mnist \
		-x 'pytorch-mnist/.venv/*'
# In SageMaker Noteebok instance's JupyterLab terminal
unzip-pytorch-mnist:
	cd SageMaker/
	unzip pytorch-mnist.zip
