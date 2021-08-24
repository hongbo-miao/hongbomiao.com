conda-create:
	conda create --name=hm-opal-client python=3.9.6
conda-env-remove:
	conda env remove --name=hm-opal-client
conda-activate:
	conda activate hm-opal-client
conda-env-list:
	conda env list
conda-install:
	conda install pytorch torchvision torchaudio --channel=pytorch
conda-list-packages:
	conda list

pip-list-packages:
	/usr/local/anaconda3/envs/hm-opal-client/bin/pip list
pip-freeze:
	/usr/local/anaconda3/envs/hm-opal-client/bin/pip freeze > requirements.txt

docker-build:
	cd .. && \
	docker build --file=hm-opal-client/Dockerfile --tag=hongbomiao/hm-opal-client:latest .
docker-push:
	docker push hongbomiao/hm-opal-client:latest
