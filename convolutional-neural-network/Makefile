poetry-lock:
	poetry lock --no-update
poetry-install:
	poetry install
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

poetry-run-dev:
	poetry run poe dev
poetry-run-dev-download-original-data:
	poetry run poe dev-download-original-data


poetry-dvc-init:
	poetry run poe dvc-init
poetry-dvc-add:
	poetry run poe dvc-add
poetry-dvc-run-prepare:
	poetry run poe dvc-run-prepare
poetry-dvc-run-main:
	poetry run poe dvc-run-main
poetry-dvc-run-main-no-exec:
	poetry run poe dvc-run-main-no-exec
poetry-dvc-repro:
	poetry run poe dvc-repro

poetry-dvc-remove-stage:
	poetry run poe dvc-remove-stage
poetry-dvc-dag:
	poetry run poe dvc-dag

poetry-dvc-remote-add:
	poetry run poe dvc-remote-add
poetry-dvc-push:
	poetry run poe dvc-push
poetry-dvc-pull:
	poetry run poe dvc-pull

aws-create-s3:
	aws s3 mb s3://hongbomiao-bucket/hm-cnn-dvc-store --region us-west-2
aws-ls:
	aws s3 ls s3://hongbomiao-bucket/hm-cnn-dvc-store
