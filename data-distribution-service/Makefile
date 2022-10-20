conda-create:
	conda create --name=hm-dds python=3.10 --yes
conda-env-remove:
	conda env remove --name=hm-dds
conda-activate:
	conda activate hm-dds
conda-deactivate:
	conda deactivate
conda-env-list:
	conda env list
conda-install:
	conda install pytorch torchvision torchaudio --channel=pytorch
conda-list-packages:
	conda list

pip-install:
	/usr/local/anaconda3/envs/hm-dds/bin/pip install wandb
pip-list-packages:
	/usr/local/anaconda3/envs/hm-dds/bin/pip list
pip-freeze:
	pip freeze > requirements.txt
