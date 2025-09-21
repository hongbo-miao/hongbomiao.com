#!/usr/bin/env bash
set -e

echo "# Install Ingress"
helm upgrade \
  ingress-nginx \
  ingress-nginx \
  --install \
  --repo=https://kubernetes.github.io/ingress-nginx \
  --namespace=ingress-nginx \
  --create-namespace \
  --values=kubernetes/manifests/ingress-nginx/helm/my-values.yaml
# helm uninstall ingress-nginx --namespace=ingress-nginx
# kubectl delete namespace ingress-nginx
echo "=================================================="
