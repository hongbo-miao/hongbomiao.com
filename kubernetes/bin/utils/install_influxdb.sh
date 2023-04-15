#!/usr/bin/env bash
set -e

echo "# Install InfluxDB"
# https://github.com/bitnami/charts/tree/main/bitnami/influxdb
helm upgrade \
  influxdb \
  influxdb \
  --install \
  --repo=https://charts.bitnami.com/bitnami \
  --namespace=hm-influxdb \
  --create-namespace \
  --values=kubernetes/manifests/influxdb/helm/my-values.yaml
# helm uninstall influxdb --namespace=hm-influxdb
# kubectl delete namespace hm-influxdb

# http://localhost:20622
# admin
# passw0rd
echo "=================================================="
