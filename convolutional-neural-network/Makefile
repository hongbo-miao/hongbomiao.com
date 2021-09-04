conda-create:
	conda create --name=hm-cnn python=3.9.6
conda-env-remove:
	conda env remove --name=hm-cnn
conda-activate:
	conda activate hm-cnn
conda-env-list:
	conda env list
conda-install:
	conda install pytorch torchvision torchaudio --channel=pytorch
conda-list-packages:
	conda list

pip-install:
	/usr/local/anaconda3/envs/hm-cnn/bin/pip install wandb
pip-list-packages:
	/usr/local/anaconda3/envs/hm-cnn/bin/pip list
pip-freeze:
	pip freeze > requirements.txt

dvc-init:
	dvc init --subdir
dvc-add:
	dvc add data/cifar-10-python.tar.gz
dvc-repro:
	dvc repro data/cifar-10-python.tar.gz.dvc
dvc-run-prepare:
	dvc run \
		--name=prepare \
		--deps=data/cifar-10-python.tar.gz \
 		--outs=data/cifar-10-batches-py/ \
		tar -xf data/cifar-10-python.tar.gz --dir=data
dvc-run-train:
	dvc run \
		--name=train \
		--deps=data/cifar-10-batches-py/ \
		python main.py --lr=0.001 --num_epochs=2

dvc-dag:
	dvc dag

aws-create-s3:
	aws s3 mb s3://hongbomiao-bucket/hm-cnn-dvc-store --region us-west-2
aws-ls:
	aws s3 ls s3://hongbomiao-bucket/hm-cnn-dvc-store
dvc-remote-add:
	dvc remote add -d hm-remote s3://hongbomiao-bucket/hm-cnn-dvc-store
dvc-push:
	dvc push
dvc-pull:
	dvc pull

run:
	/usr/local/anaconda3/envs/hm-cnn/bin/python main.py --lr=0.001 --num_epochs=2
