poetry-env-use:
	poetry env use python3.11
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root

poetry-run-test:
	poetry run poe test
poetry-run-test-coverage:
	poetry run poe test-coverage

sky-launch:
	poetry run sky launch skypilot.yaml \
		--cluster=hm-skypilot-cluster \
		--idle-minutes-to-autostop=10 \
		--yes
sky-stop:
	poetry run sky stop hm-skypilot-cluster --yes
sky-terminate:
	poetry run sky down hm-skypilot-cluster --yes

sky-check:
	poetry run sky check
sky-status:
	poetry run sky status
sky-ssh:
	poetry run ssh hm-skypilot-cluster

sky-show-gpus:
	poetry run sky show-gpus
sky-show-gpus-price:
	poetry run sky show-gpus --all
sky-show-gpus-price-a100-80gb-8:
	poetry run sky show-gpus A100-80GB:8
