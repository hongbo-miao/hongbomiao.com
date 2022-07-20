#!/usr/bin/env bash
set -e

echo "# Install Spark"
# https://github.com/bitnami/charts/tree/master/bitnami/spark
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm install \
  spark \
  bitnami/spark \
  --namespace=hm-spark \
  --create-namespace

# Delete:
# helm uninstall spark --namespace=hm-spark
echo "=================================================="
