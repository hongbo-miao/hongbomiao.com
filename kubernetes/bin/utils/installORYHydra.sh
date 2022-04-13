#!/usr/bin/env bash
set -e

echo "# Install ORY Hydra"
kubectl apply --filename=kubernetes/config/ory-hydra/hm-ory-hydra-namespace.yaml
helm repo add ory https://k8s.ory.sh/helm/charts
helm repo update
helm install ory-hydra \
  --namespace=hm-ory-hydra \
  --values=ory-hydra/ory-hydra.yaml \
  ory/hydra

# Delete:
# helm uninstall ory-hydra --namespace=hm-ory-hydra
echo "=================================================="
