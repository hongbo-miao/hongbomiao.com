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
  --values=kubernetes/manifests/prefect/helm/prefect-orion/my-values.yaml

# Delete:
# helm uninstall prefect-orion --namespace=hm-prefect
echo "=================================================="

echo "# Install Prefect Agent"
helm install \
  prefect-agent \
  prefect/prefect-agent \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/prefect-agent/my-values.yaml

# Delete:
# helm uninstall prefect-agent --namespace=hm-prefect
echo "=================================================="
