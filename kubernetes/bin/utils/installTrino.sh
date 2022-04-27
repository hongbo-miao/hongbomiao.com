#!/usr/bin/env bash
set -e

echo "# Install Trino"
# https://github.com/trinodb/charts
kubectl apply --filename=kubernetes/manifests/trino/hm-trino-namespace.yaml
helm repo add trino https://trinodb.github.io/charts
helm install \
  trino \
  trino/trino \
  --namespace=hm-trino \
  --values=kubernetes/manifests/trino/helm/my-values.yaml

# Username: admin

# Dry run:
# helm install \
#   trino \
#   trino/trino \
#   --namespace=hm-trino \
#   --values=kubernetes/manifests/trino/helm/my-values.yaml \
#   --dry-run

# Upgrade:
# helm upgrade \
#   trino \
#   trino/trino \
#   --namespace=hm-trino \
#   --values=kubernetes/manifests/trino/helm/my-values.yaml

# Delete:
# helm uninstall trino --namespace=hm-trino
echo "=================================================="
