#!/usr/bin/env bash
set -e

echo "# Install Traefik Ingress Controller"
helm upgrade \
  traefik \
  traefik \
  --install \
  --repo=https://traefik.github.io/charts \
  --namespace=hm-traefik \
  --create-namespace
# helm uninstall traefik --namespace=hm-traefik
# kubectl delete namespace hm-traefik
echo "=================================================="


echo "# Install Traefik Ingress for the namespace hm-prefect"
kubectl apply --filename=kubernetes/manifests/traefik/hm-prefect-traefik-ingress.yaml
# kubectl delete --filename=kubernetes/manifests/traefik/hm-prefect-traefik-ingress.yaml
echo "=================================================="
