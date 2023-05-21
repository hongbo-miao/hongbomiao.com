#!/usr/bin/env bash
set -e

# Creating an environment
conda create --name=hm-gnn python=3.8 --yes
conda activate hm-gnn

# Install PyTorch Geometric
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
TORCH="1.8.0"
CUDA="cpu" # CUDA="cu111"
pip install torch-scatter --find-links=https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse --find-links=https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster --find-links=https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv --find-links=https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric

# Install requirements
pip install --requirement=requirements.txt
