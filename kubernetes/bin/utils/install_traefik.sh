#!/usr/bin/env bash
set -e

echo "# Install Traefik Ingress Controller"
helm repo add traefik https://traefik.github.io/charts
helm install \
  traefik \
  traefik/traefik \
  --namespace=hm-traefik \
  --create-namespace

# Delete:
# helm uninstall traefik --namespace=hm-traefik
echo "=================================================="


echo "# Install Traefik Ingress for the namespace hm-prefect"
kubectl apply --filename=kubernetes/manifests/traefik/hm-prefect-traefik-ingress.yaml
# kubectl delete --filename=kubernetes/manifests/traefik/hm-prefect-traefik-ingress.yaml
echo "=================================================="
