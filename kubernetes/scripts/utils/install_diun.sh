#!/usr/bin/env bash
set -e

echo "# Install Diun"
helm upgrade \
  diun \
  diun \
  --install \
  --repo=https://nicholaswilde.github.io/helm-charts \
  --namespace=hm-dgraph \
  --create-namespace \
  --values=kubernetes/manifests/diun/helm/my-values.yaml
# helm uninstall diun --namespace=hm-diun
# kubectl delete namespace hm-diun
echo "=================================================="
