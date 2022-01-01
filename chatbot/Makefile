conda-create:
	conda create --name=hm-chatbot python=3.8
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

# https://rasa.com/docs/rasa/command-line-interface
rasa-init:
	rasa init
rasa-train:
	rasa train
rasa-test:
	rasa test
rasa-shell:
	rasa shell
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

docker-build:
	cd .. && \
	docker build --file=chatbot/Dockerfile --tag=hongbomiao/hm-chatbot:latest .
docker-push:
	docker push hongbomiao/hm-chatbot:latest
