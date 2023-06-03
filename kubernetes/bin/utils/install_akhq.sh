#!/usr/bin/env bash
set -e

echo "# Install AKHQ"
# https://akhq.io/docs/installation.html#running-in-kubernetes-using-a-helm-chart
helm upgrade \
  akhq \
  akhq \
  --install \
  --repo=https://akhq.io \
  --namespace=hm-akhq \
  --create-namespace \
  --values=kubernetes/manifests/akhq/helm/my-values.yaml
# helm uninstall akhq --namespace=hm-akhq
# kubectl delete namespace hm-akhq
echo "=================================================="

helm upgrade \
  akhq \
  akhq \
  --namespace=hm-akhq \
  --values=kubernetes/manifests/akhq/helm/my-values.yaml
