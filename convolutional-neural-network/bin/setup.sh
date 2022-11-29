#!/usr/bin/env bash
set -e

# Creating an environment
conda create --name=hm-cnn python=3.9 --yes
conda activate hm-cnn

# Install requirements
conda install pytorch torchvision torchaudio --channel=pytorch --yes
conda install dvc dvc-s3 --channel=conda-forge --yes
conda install pandas --yes
conda install tabulate --yes

pip install --requirement=requirements.txt
