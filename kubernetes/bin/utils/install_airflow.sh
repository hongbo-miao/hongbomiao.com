#!/usr/bin/env bash
set -e

echo "# Install Airflow"
# https://airflow.apache.org/docs/helm-chart/stable/index.html
helm repo add apache-airflow https://airflow.apache.org
helm repo update apache-airflow
helm install \
  airflow \
  apache-airflow/airflow \
  --namespace=hm-airflow \
  --create-namespace \
  --values=kubernetes/manifests/airflow/helm/my-values.yaml

# Upgrade:
# helm upgrade \
#   airflow \
#   apache-airflow/airflow \
#   --namespace=hm-airflow \
#   --values=kubernetes/manifests/airflow/helm/my-values.yaml

# Delete:
# helm uninstall airflow --namespace=hm-airflow
echo "=================================================="
