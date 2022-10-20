#!/usr/bin/env bash
set -e

# Creating an environment
conda create --name=hm-quantum python=3.9 --yes
conda activate hm-quantum

# Install requirements
pip install qiskit
pip install 'qiskit[visualization]'
