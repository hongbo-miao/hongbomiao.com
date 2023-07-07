skypilot:
	cd .. && \
	docker run \
		--interactive \
		--tty \
		--rm \
		--volume="$(pwd):/workspace" \
		--volume="$HOME/.sky:/root/.sky:rw" \
		--volume="$HOME/.aws:/root/.aws:rw" \
		--volume="$HOME/.config/gcloud:/root/.config/gcloud:rw" \
		--volume="$HOME/.azure:/root/.azure:rw" \
		docker.io/berkeleyskypilot/skypilot:latest \
		/bin/bash

skypilot-launch:
	sky launch --cluster=hm-skypilot-cluster skypilot.yaml --idle-minutes-to-autostop=10
skypilot-stop:
	sky stop hm-skypilot-cluster
skypilot-terminate:
	sky down hm-skypilot-cluster

skypilot-check:
	sky check
skypilot-status:
	sky status
skypilot-show-gpus:
	sky show-gpus
skypilot-ssh:
	ssh hm-skypilot-cluster
