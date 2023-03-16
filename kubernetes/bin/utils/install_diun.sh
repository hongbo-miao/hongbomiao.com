#!/usr/bin/env bash
set -e

echo "# Install Diun"
helm repo add nicholaswilde https://nicholaswilde.github.io/helm-charts
helm install my-diun \
  nicholaswilde/diun \
  --namespace=hm-diun \
  --create-namespace \
  --values=kubernetes/manifests/diun/helm/my-values.yaml

# Delete:
# helm uninstall diun --namespace=hm-diun
echo "=================================================="
