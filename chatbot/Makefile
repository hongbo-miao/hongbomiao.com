# Setup
conda-create:
	conda create --name=hm-chatbot python=3.8 --yes
conda-env-remove:
	conda env remove --name=hm-chatbot
conda-activate:
	conda activate hm-chatbot
conda-deactivate:
	conda deactivate
conda-env-list:
	conda env list
conda-list-packages:
	conda list

pip-install:
	pip install rasa

# Rasa
# https://rasa.com/docs/rasa/command-line-interface
rasa-init:
	rasa init
rasa-train:
	rasa train --num-threads=8
rasa-test:
	rasa test
rasa-test-story:
	rasa test --stories tests/test_stories.yml
rasa-shell:
	rasa shell
rasa-interactive:
	rasa interactive
rasa-start-rasa-server:
	rasa run --enable-api --port=39143
rasa-start-action-server:
	rasa run actions
rasa-visualize:
	rasa visualize
rasa-help:
	rasa --help
rasa-telemetry-disable:
	rasa telemetry disable
rasa-x:
	rasa x

# Docker
docker-build:
	cd .. && \
	docker build --file=chatbot/Dockerfile --tag=hongbomiao/hm-rasa-action-server:latest .
docker-push:
	docker push hongbomiao/hm-rasa-action-server:latest
