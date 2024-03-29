#!/usr/bin/env bash
set -e

echo "# Install Telegraf"
kubectl apply --filename=kubernetes/manifests/telegraf/hm-telegraf-namespace.yaml

INFLUXDB_API_TOKEN="xxx"
kubectl create secret generic hm-telegraf-secret \
  --namespace=hm-telegraf \
  --from-literal="influxdb_api_token=${INFLUXDB_API_TOKEN}"
kubectl apply --filename=kubernetes/manifests/telegraf
# kubectl delete --filename=kubernetes/manifests/telegraf
echo "=================================================="
