#!/usr/bin/env bash
set -e

# Creating an environment
conda create --name=hm-dds python=3.10 --yes
conda activate hm-dds

# Install requirements
pip install --requirement=requirements.txt

# Build and install rti
cd submodules/connextdds-py
git checkout tags/v0.1.5
python configure.py --nddshome=/Applications/rti_connext_dds-6.1.1 --jobs=8 arm64Darwin20clang12.0
pip intall .
rm -f rti-0.1.5-cp310-cp310-macosx_12_0_arm64.whl
git checkout master
cd ../..
