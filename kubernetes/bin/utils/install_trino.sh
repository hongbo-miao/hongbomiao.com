#!/usr/bin/env bash
set -e

echo "# Install Trino"
# https://github.com/trinodb/charts
# helm upgrade \
#   trino \
#   trino \
#   --install \
#   --repo=https://trinodb.github.io/charts \
#   --namespace=hm-trino \
#   --create-namespace \
#   --values=kubernetes/manifests/trino/helm/my-values.yaml

helm repo add trino https://trinodb.github.io/charts
helm install trino trino/trino \
  --namespace=hm-trino \
  --create-namespace \
  --values=kubernetes/manifests/trino/helm/my-values.yaml

# helm uninstall trino --namespace=hm-trino
# kubectl delete namespace hm-trino

# Username: admin
echo "=================================================="
