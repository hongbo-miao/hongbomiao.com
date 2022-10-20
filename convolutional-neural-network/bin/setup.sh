#!/usr/bin/env bash
set -e

# Creating an environment
conda create --name=hm-cnn python=3.9 --yes
conda activate hm-cnn

# Install requirements
conda install pytorch torchvision torchaudio --channel=pytorch
conda install dvc dvc-s3 --channel=conda-forge
conda install pandas
conda install tabulate

pip install --requirement=requirements.txt
