poetry-env-use:
	poetry env use 3.10
poetry-update-lock-file:
	poetry lock --no-update
poetry-install:
	poetry install --no-root
poetry-add:
	poetry add xxx
poetry-add-dev:
	poetry add xxx --group=dev

poetry-shell:
	poetry shell

sky-launch:
	sky launch skypilot.yaml \
		--cluster=hm-skypilot-cluster \
		--idle-minutes-to-autostop=10 \
		--yes
sky-stop:
	sky stop hm-skypilot-cluster --yes
sky-terminate:
	sky down hm-skypilot-cluster --yes

sky-check:
	sky check
sky-status:
	sky status
sky-show-gpus:
	sky show-gpus
sky-ssh:
	ssh hm-skypilot-cluster
