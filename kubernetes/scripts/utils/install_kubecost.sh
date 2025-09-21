#!/usr/bin/env bash
set -e

echo "# Install Kubecost"
helm upgrade \
  kubecost \
  cost-analyzer \
  --install \
  --repo=https://kubecost.github.io/cost-analyzer \
  --namespace=hm-kubecost \
  --create-namespace \
  --values=kubernetes/manifests/dgraph/helm/my-values.yaml
# helm uninstall kubecost --namespace=hm-kubecost
# kubectl delete namespace hm-kubecost
echo "=================================================="
