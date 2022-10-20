conda-create:
	conda create --name=hm-opal-client python=3.9 --yes
conda-env-remove:
	conda env remove --name=hm-opal-client
conda-activate:
	conda activate hm-opal-client

pip-freeze:
	/usr/local/anaconda3/envs/hm-opal-client/bin/pip freeze > requirements.txt

docker-build:
	cd .. && \
	docker build --file=hm-opal-client/Dockerfile --tag=ghcr.io/hongbo-miao/hm-opal-client:latest .
docker-push:
	docker push ghcr.io/hongbo-miao/hm-opal-client:latest
