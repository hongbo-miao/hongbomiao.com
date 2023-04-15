#!/usr/bin/env bash
set -e

echo "# Install Airflow"
# https://airflow.apache.org/docs/helm-chart/stable/index.html
helm upgrade \
  airflow \
  airflow \
  --install \
  --repo=https://airflow.apache.org \
  --namespace=hm-airflow \
  --create-namespace \
  --values=kubernetes/manifests/airflow/helm/my-values.yaml
# helm uninstall airflow --namespace=hm-airflow
# kubectl delete namespace hm-airflow
echo "=================================================="
