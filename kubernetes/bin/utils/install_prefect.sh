#!/usr/bin/env bash
set -e

source kubernetes/bin/utils/install_cloudflared.sh

echo "# Install Prefect Server"
helm repo add prefect https://prefecthq.github.io/prefect-helm
# helm repo remove prefect
# helm repo update prefect
# helm search repo prefect
helm install \
  prefect-server \
  prefect/prefect-server \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/prefect-server/my-values.yaml

# Delete:
# helm uninstall prefect-server --namespace=hm-prefect
echo "=================================================="

echo "# Create hm-kubernetes profile"
prefect profile create hm-kubernetes
prefect profile use hm-kubernetes
# prefect profile use default
# prefect config view
prefect config set PREFECT_API_URL=https://tunnel.hongbomiao.com/api
echo "=================================================="

echo "# Build hm-prefect-print-platform"
docker build --file=hm-prefect/workflows/print-platform/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-print-platform:latest .
docker push ghcr.io/hongbo-miao/hm-prefect-print-platform:latest
echo "=================================================="

# hm-prefect-print-platform
cd hm-prefect/workflows/print-platform

echo "# Start the workflow"
poetry run poe add-kubernetes-job-block
poetry run poe build -- --work-queue=hm-local-queue
# poetry run poe build -- --params=$(cat params.json | jq -c .) --work-queue=hm-local-queue
# poetry run poe build -- --work-queue=hm-kubernetes-queue
poetry run poe run

# Delete:
# kubectl delete jobs --all --namespace=hm-prefect
echo "=================================================="

echo "# Start Prefect Agent in local"
poetry run poe prefect-agent-start -- --work-queue=hm-local-queue
echo "=================================================="

echo "# Start Prefect Agent in Kubernetes"
helm install \
  prefect-agent \
  prefect/prefect-agent \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/prefect-agent/my-values.yaml

# Delete:
# helm uninstall prefect-agent --namespace=hm-prefect
echo "=================================================="
