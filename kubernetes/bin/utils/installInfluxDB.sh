#!/usr/bin/env bash
set -e

echo "# Install InfluxDB"
# https://github.com/bitnami/charts/tree/master/bitnami/influxdb
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
helm install \
  influxdb \
  bitnami/influxdb \
  --namespace=hm-influxdb \
  --create-namespace \
  --values=kubernetes/manifests/influxdb/helm/my-values.yaml

# Delete:
# helm uninstall influxdb --namespace=hm-influxdb
echo "=================================================="
