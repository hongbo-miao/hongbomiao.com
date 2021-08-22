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

run:
	/usr/local/anaconda3/envs/hm-cnn/bin/python main.py --lr=0.001 --num_epochs=2
