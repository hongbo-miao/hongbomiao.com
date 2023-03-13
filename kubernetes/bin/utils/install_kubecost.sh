#!/usr/bin/env bash
set -e

echo "# Install Kubecost"
helm repo add kubecost https://kubecost.github.io/cost-analyzer
helm install \
  kubecost \
  kubecost/cost-analyzer \
  --namespace=hm-kubecost \
  --create-namespace

# Delete:
# helm uninstall kubecost --namespace=hm-kubecost
echo "=================================================="
