skypilot:
	cd .. && \
	docker run \
		--name=skypilot \
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
	sky launch skypilot.yaml \
		--cluster=hm-skypilot-cluster \
		--idle-minutes-to-autostop=10 \
		--yes
skypilot-stop:
	sky stop hm-skypilot-cluster --yes
skypilot-terminate:
	sky down hm-skypilot-cluster --yes

skypilot-check:
	sky check
skypilot-status:
	sky status
skypilot-show-gpus:
	sky show-gpus
skypilot-ssh:
	ssh hm-skypilot-cluster
