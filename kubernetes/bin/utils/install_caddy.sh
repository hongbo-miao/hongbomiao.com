#!/usr/bin/env bash
set -e

echo "# Install Caddy Ingress Controller"
helm upgrade \
  caddy \
  caddy-ingress-controller \
  --install \
  --repo=https://caddyserver.github.io/ingress \
  --namespace=hm-caddy \
  --create-namespace \
  --values=kubernetes/manifests/caddy/helm/my-values.yaml
# helm uninstall caddy --namespace=hm-caddy
# kubectl delete namespace hm-caddy
echo "=================================================="

echo "# Install Caddy Ingress for the namespace hm-prefect"
kubectl apply --filename=kubernetes/manifests/caddy/hm-prefect-caddy-ingress.yaml
# kubectl delete --filename=kubernetes/manifests/caddy/hm-prefect-caddy-ingress.yaml
echo "=================================================="
