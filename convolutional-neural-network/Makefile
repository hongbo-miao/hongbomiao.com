conda-create:
	conda create --name=hm-cnn python=3.9 --yes
conda-env-remove:
	conda env remove --name=hm-cnn
conda-activate:
	conda activate hm-cnn
conda-deactivate:
	conda deactivate
conda-env-list:
	conda env list
conda-install:
	conda install pytorch torchvision torchaudio --channel=pytorch --yes
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
	dvc add data/raw/cifar-10-python.tar.gz
dvc-run-prepare:
	dvc run \
		--name=prepare \
		--deps=data/raw/cifar-10-python.tar.gz \
 		--outs=data/processed/cifar-10-batches-py/ \
		tar -xf data/raw/cifar-10-python.tar.gz --dir=data/processed
dvc-run-main:
	dvc run \
		--name=main \
		--deps=data/processed/cifar-10-batches-py/ \
		--deps=model/ \
		--deps=main.py \
		--deps=train.py \
		--deps=evaluate.py \
		--params=lr,train.epochs \
 		--outs=output/models/model.pt \
		python main.py
dvc-run-main-no-exec:
	dvc run \
		--no-exec \
		--name=main \
		--deps=data/cifar-10-batches-py/ \
		--deps=model/ \
		--deps=main.py \
		--deps=train.py \
		--deps=evaluate.py \
		--params=lr,train.epochs \
		python main.py
dvc-repro:
	dvc repro data/raw/cifar-10-python.tar.gz.dvc

dvc-remove-stage:
	dvc remove STAGE_NAME
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
	/usr/local/anaconda3/envs/hm-cnn/bin/python main.py --should_download_original_data=True
