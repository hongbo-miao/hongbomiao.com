poetry-env-use:
	poetry env use python3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-dev:
	export WANDB_MODE=offline && \
	poetry run poe dev
poetry-run-dev-should-download-original-data:
	export WANDB_MODE=offline && \
	poetry run poe dev -- --should_download_original_data=True
poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

poetry-dvc-init:
	poetry run poe dvc-init
poetry-dvc-add:
	poetry run poe dvc-add -- data/raw/cifar-10-python.tar.gz
poetry-dvc-run-prepare:
	poetry run poe dvc-run -- --name=prepare --deps=data/raw/cifar-10-python.tar.gz --outs=data/processed/cifar-10-batches-py/ tar -x -f data/raw/cifar-10-python.tar.gz -C data/processed
poetry-dvc-run-main:
	poetry run poe dvc-run -- --name=main --deps=data/processed/cifar-10-batches-py/ --deps=model/ --deps=main.py --deps=train.py --deps=evaluate.py --params=lr,train.epochs --outs=output/models/model.pt python main.py
poetry-dvc-run-main-no-exec:
	poetry run poe dvc-run -- --no-exec --name=main --deps=data/processed/cifar-10-batches-py/ --deps=model/ --deps=main.py --deps=train.py --deps=evaluate.py --params=lr,train.epochs --outs=output/models/model.pt python main.py
poetry-dvc-repro:
	poetry run poe dvc-repro

poetry-dvc-remove:
	poetry run poe dvc-remove -- STAGE_NAME
poetry-dvc-dag:
	poetry run poe dvc-dag

poetry-dvc-remote-add:
	poetry run poe dvc-remote-add -- -d hm-remote s3://hm-production-bucket/hm-cnn-dvc-store
poetry-dvc-push:
	poetry run poe dvc-push
poetry-dvc-pull:
	poetry run poe dvc-pull

aws-create-s3:
	aws s3 mb s3://hm-production-bucket/hm-cnn-dvc-store --region us-west-2
aws-ls:
	aws s3 ls s3://hm-production-bucket/hm-cnn-dvc-store
