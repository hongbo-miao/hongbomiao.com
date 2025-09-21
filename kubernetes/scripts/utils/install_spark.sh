#!/usr/bin/env bash
set -e

echo "# Install Spark"
# https://github.com/bitnami/charts/tree/main/bitnami/spark
helm upgrade \
  spark \
  spark \
  --install \
  --repo=https://charts.bitnami.com/bitnami \
  --namespace=hm-spark \
  --create-namespace \
  --values=kubernetes/manifests/spark/helm/my-values.yaml
# helm uninstall spark --namespace=hm-spark
# kubectl delete namespace hm-spark
echo "=================================================="
