#!/usr/bin/env bash

set -e

# Creating an environment
conda create --name hm-cnn python=3.9.6
conda activate hm-cnn

# Install requirements
conda install pytorch torchvision torchaudio -c pytorch
conda install pandas
conda install tabulate

pip install --requirement requirements.txt
