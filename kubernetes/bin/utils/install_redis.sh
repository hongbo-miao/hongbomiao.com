#!/usr/bin/env bash
set -e

echo "# Install Redis"
kubectl apply --filename=kubernetes/manifests/redis/hm-redis-namespace.yaml
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install \
  redis \
  bitnami/redis \
  --namespace=hm-redis \
  --values=kubernetes/manifests/redis/helm/my-values.yaml

# Delete:
# helm uninstall redis --namespace=hm-redis
echo "=================================================="
