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

sky-launch:
	uv run sky launch skypilot.yaml \
		--cluster=hm-skypilot-cluster \
		--idle-minutes-to-autostop=10 \
		--yes
sky-stop:
	uv run sky stop hm-skypilot-cluster --yes
sky-terminate:
	uv run sky down hm-skypilot-cluster --yes

sky-check:
	uv run sky check
sky-status:
	uv run sky status
sky-ssh:
	uv run ssh hm-skypilot-cluster

sky-show-gpus:
	uv run sky show-gpus
sky-show-gpus-price:
	uv run sky show-gpus --all
sky-show-gpus-price-a100-80gb-8:
	uv run sky show-gpus A100-80GB:8
