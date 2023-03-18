#!/usr/bin/env bash
set -e

echo "# Install InfluxDB"
# https://github.com/bitnami/charts/tree/main/bitnami/influxdb
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update bitnami
helm install \
  influxdb \
  bitnami/influxdb \
  --namespace=hm-influxdb \
  --create-namespace \
  --values=kubernetes/manifests/influxdb/helm/my-values.yaml

# Delete:
# helm uninstall influxdb --namespace=hm-influxdb

# http://localhost:20622
# admin
# passw0rd
echo "=================================================="
