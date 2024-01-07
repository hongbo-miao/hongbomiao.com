#!/usr/bin/env bash
set -e

echo "# Install JupyterLab-scoped dependencies"
PYTHON_VERSION=3.11.7
sudo /emr/notebook-env/bin/conda create --name="python${PYTHON_VERSION}" python=${PYTHON_VERSION} --yes
sudo "/emr/notebook-env/envs/python${PYTHON_VERSION}/bin/python" -m pip install \
  apache-sedona[spark]==1.5.0 \
  attrs==23.1.0 \
  descartes==1.1.0 \
  ipykernel==6.28.0 \
  matplotlib==3.8.2 \
  pandas==2.1.4 \
  shapely==2.0.2

echo "# Add JupyterLab kernel"
sudo "/emr/notebook-env/envs/python${PYTHON_VERSION}/bin/python" -m ipykernel install --name="python${PYTHON_VERSION}"
