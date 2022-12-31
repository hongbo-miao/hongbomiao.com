#!/usr/bin/env bash
set -e

echo "# Install Caddy Ingress Controller"
helm install caddy \
  caddy-ingress-controller \
  --namespace=hm-caddy \
  --create-namespace \
  --repo=https://caddyserver.github.io/ingress \
  --values=kubernetes/manifests/caddy/helm/my-values.yaml

# Delete:
# helm uninstall caddy --namespace=hm-caddy
echo "=================================================="


echo "# Install Caddy Ingress for the namespace hm-prefect"
kubectl apply --filename=kubernetes/manifests/caddy/hm-prefect-caddy-ingress.yaml
# kubectl delete --filename=kubernetes/manifests/caddy/hm-prefect-caddy-ingress.yaml
echo "=================================================="
