#!/usr/bin/env bash
set -e

echo "# Install Polaris"
# https://polaris.docs.fairwinds.com/dashboard
helm upgrade \
  polaris \
  polaris \
  --install \
  --repo=https://charts.fairwinds.com/stable \
  --namespace=hm-polaris \
  --create-namespace \
  --values=kubernetes/manifests/polaris/helm/my-values.yaml
# helm uninstall polaris --namespace=hm-polaris
# kubectl delete namespace hm-polaris
echo "=================================================="
