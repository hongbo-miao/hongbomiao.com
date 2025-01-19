uv-install-python:
	uv python install
uv-update-lock-file:
	uv lock
uv-install-dependencies:
	uv sync --dev

uv-run-dev:
	uv run poe dev
uv-run-dev-wandb-offline:
	export WANDB_MODE=offline && \
	uv run poe dev
uv-run-dev-download-original-data:
	uv run poe dev --should_download_original_data=True
uv-run-test:
	uv run poe test
uv-run-test-coverage:
	uv run poe test-coverage

poetry-dvc-init:
	uv run poe dvc-init
poetry-dvc-add:
	uv run poe dvc-add -- data/raw/cifar-10-python.tar.gz
poetry-dvc-run-prepare:
	uv run poe dvc-run -- --name=prepare --deps=data/raw/cifar-10-python.tar.gz --outs=data/processed/cifar-10-batches-py/ tar -x -f data/raw/cifar-10-python.tar.gz -C data/processed
poetry-dvc-run-main:
	uv run poe dvc-run -- --name=main --deps=data/processed/cifar-10-batches-py/ --deps=model/ --deps=main.py --deps=train.py --deps=evaluate.py --params=lr,train.epochs --outs=output/models/model.pt python main.py
poetry-dvc-run-main-no-exec:
	uv run poe dvc-run -- --no-exec --name=main --deps=data/processed/cifar-10-batches-py/ --deps=model/ --deps=main.py --deps=train.py --deps=evaluate.py --params=lr,train.epochs --outs=output/models/model.pt python main.py
poetry-dvc-repro:
	uv run poe dvc-repro

poetry-dvc-remove:
	uv run poe dvc-remove -- STAGE_NAME
poetry-dvc-dag:
	uv run poe dvc-dag

poetry-dvc-remote-add:
	uv run poe dvc-remote-add -- -d hm-remote s3://hm-production-bucket/hm-cnn-dvc-store
poetry-dvc-push:
	uv run poe dvc-push
poetry-dvc-pull:
	uv run poe dvc-pull

aws-create-s3:
	aws s3 mb s3://hm-production-bucket/hm-cnn-dvc-store --region us-west-2
aws-ls:
	aws s3 ls s3://hm-production-bucket/hm-cnn-dvc-store
