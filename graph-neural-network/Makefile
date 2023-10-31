poetry-env-use:
	poetry env use 3.8
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-dev:
	poetry run poe dev -- --dataset=ogbg-molhiv --gnn=gcn
poetry-run-dev-debug:
	export WANDB_MODE=offline && \
	poetry run poe dev -- --dataset=ogbg-molhiv --gnn=gcn --feature=simple --epochs=2
