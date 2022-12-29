#!/usr/bin/env bash
set -e

echo "# Install Prefect Orion"
helm repo add prefect https://prefecthq.github.io/prefect-helm
# helm search repo prefect
helm install \
  prefect-orion \
  prefect/prefect-orion \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/my-values.yaml

# Delete:
# helm uninstall prefect-orion --namespace=hm-prefect
echo "=================================================="
