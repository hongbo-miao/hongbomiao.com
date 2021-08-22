#!/usr/bin/env bash

set -e

# Creating an environment
conda create --name hm-cnn python=3.9.6
conda activate hm-cnn

# Install PyTorch
conda install pytorch torchvision torchaudio -c pytorch

# Install requirements
pip install --requirement requirements.txt
