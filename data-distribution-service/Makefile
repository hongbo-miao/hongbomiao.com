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
	conda install pytorch torchvision torchaudio --channel=pytorch --yes
conda-list-packages:
	conda list

pip-install:
	pip install xxx
pip-list-packages:
	pip list
pip-freeze:
	pip freeze > requirements.txt

rti-convert:
	rtiddsgen -convertToXml hm_message.idl -d .

python-publisher:
	python publish.py
python-subscribe:
	python subscribe.py
