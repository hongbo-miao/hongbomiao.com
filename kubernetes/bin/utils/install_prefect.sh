#!/usr/bin/env bash
set -e

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

kubectl port-forward service/prefect-server --namespace=hm-prefect 4200:4200 &
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

echo "# Create hm-kubernetes profile"
prefect profile create hm-kubernetes
prefect profile use hm-kubernetes
# prefect profile use default
# prefect config view
prefect config set PREFECT_API_URL=http://localhost:4200/api
echo "=================================================="

echo "# Build hm-prefect-print-platform"
docker build --file=hm-prefect/build/print_platform/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-print-platform:latest .
docker push ghcr.io/hongbo-miao/hm-prefect-print-platform:latest
echo "=================================================="

echo "# Start hm-prefect-print-platform"
poetry run poe add-kubernetes-job-block-print-platform
poetry run poe build-kubernetes-print-platform
poetry run poe run-print-platform
poetry run poe prefect-agent-start
# kubectl delete jobs --all --namespace=hm-prefect
echo "=================================================="
