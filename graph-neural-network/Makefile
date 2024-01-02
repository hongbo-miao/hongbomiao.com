poetry-env-use:
	poetry env use python3.8
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-dev:
	poetry run poe dev -- --dataset=ogbg-molhiv --gnn=gcn
poetry-run-dev-debug:
	export WANDB_MODE=offline && \
	poetry run poe dev -- --dataset=ogbg-molhiv --gnn=gcn --feature=simple --epochs=2
poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage
