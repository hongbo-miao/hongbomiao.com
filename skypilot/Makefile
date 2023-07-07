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
		public.ecr.aws/a9w6z7w5/sky:latest \
		/bin/bash

skypilot-launch:
	sky launch --cluster=hm-classify-fashion-mnist-cluster skypilot.yaml --idle-minutes-to-autostop=10
skypilot-stop:
	sky stop hm-classify-fashion-mnist-cluster
skypilot-terminate:
	sky down hm-classify-fashion-mnist-cluster

skypilot-check:
	sky check
skypilot-status:
	sky status
skypilot-show-gpus:
	sky show-gpus
