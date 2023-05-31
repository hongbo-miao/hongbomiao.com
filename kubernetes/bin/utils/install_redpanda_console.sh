#!/usr/bin/env bash
set -e

echo "# Install Redpanda Console"
# https://github.com/redpanda-data/console
helm upgrade \
  redpanda-console \
  console \
  --install \
  --repo=https://charts.redpanda.com \
  --namespace=hm-redpanda-console \
  --create-namespace \
  --values=kubernetes/manifests/redpanda-console/helm/my-values.yaml
# helm uninstall redpanda-console --namespace=hm-redpanda-console
# kubectl delete namespace hm-redpanda-console
echo "=================================================="
