#!/usr/bin/env bash
set -e

echo "# Install Dgraph"
# https://dgraph.io/docs/deploy/kubernetes/#using-helm-chart
helm upgrade \
  dgraph \
  dgraph \
  --install \
  --repo=https://charts.dgraph.io \
  --namespace=hm-dgraph \
  --create-namespace \
  --values=kubernetes/manifests/dgraph/helm/my-values.yaml
# helm uninstall dgraph --namespace=hm-dgraph
# kubectl delete namespace hm-dgraph
echo "=================================================="
