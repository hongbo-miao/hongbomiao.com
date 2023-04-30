#!/usr/bin/env bash
set -e

echo "# Install InfluxDB"
# https://github.com/bitnami/charts/tree/main/bitnami/influxdb
helm upgrade \
  influxdb \
  oci://registry-1.docker.io/bitnamicharts/influxdb \
  --install \
  --namespace=hm-influxdb \
  --create-namespace \
  --values=kubernetes/manifests/influxdb/helm/my-values.yaml
# helm uninstall influxdb --namespace=hm-influxdb
# kubectl delete namespace hm-influxdb

# http://localhost:20622
# admin
# passw0rd
echo "=================================================="
