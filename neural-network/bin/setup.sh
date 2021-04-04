#!/usr/bin/env bash

set -e

# Creating an environment
conda create --name hm-neural-network python=3.8
conda activate hm-neural-network

# Install PyTorch Geometric
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
TORCH="1.8.0"
CUDA="cpu" # CUDA="cu111"
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

# Install requirements
pip install --requirement requirements.txt
