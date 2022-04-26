#!/usr/bin/env bash
set -e

echo "# Install Dgraph"
# https://dgraph.io/docs/deploy/kubernetes/#using-helm-chart
kubectl apply --filename=kubernetes/manifests/dgraph/hm-dgraph-namespace.yaml
helm repo add dgraph https://charts.dgraph.io
helm install \
  dgraph \
  dgraph/dgraph \
  --namespace=hm-dgraph \
  --values=kubernetes/manifests/dgraph/helm/my-values.yaml

# Delete:
# helm uninstall dgraph --namespace=hm-dgraph
echo "=================================================="
