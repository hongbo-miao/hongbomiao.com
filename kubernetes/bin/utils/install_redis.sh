#!/usr/bin/env bash
set -e

echo "# Install Redis"
helm upgrade \
  redis \
  redis \
  --install \
  --repo=https://charts.bitnami.com/bitnami \
  --namespace=hm-redis \
  --create-namespace \
  --values=kubernetes/manifests/redis/helm/my-values.yaml
# helm uninstall redis --namespace=hm-redis
# kubectl delete namespace hm-redis
echo "=================================================="
