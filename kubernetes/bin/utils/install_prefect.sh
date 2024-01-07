#!/usr/bin/env bash
set -e

echo "# Install cloudflared"
source kubernetes/bin/utils/install_cloudflared.sh
echo "=================================================="

echo "# Install Postgres"
kubectl apply --filename=kubernetes/manifests/postgres
psql postgresql://admin@localhost:5432/postgres --command="create database hm_prefect_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database hm_prefect_db to admin;"
echo "=================================================="

echo "# Install Prefect Server"
kubectl create namespace hm-prefect
kubectl create secret generic hm-prefect-postgres-secret \
  --namespace=hm-prefect \
  --from-literal="connection-string=postgresql+asyncpg://admin:passw0rd@postgres-service.hm-postgres.svc:5432/hm_prefect_db"
# kubectl delete secret hm-prefect-postgres-secret --namespace=hm-prefect

helm upgrade \
  prefect-server \
  prefect-server \
  --install \
  --repo=https://prefecthq.github.io/prefect-helm \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/prefect-server/my-values.yaml
# helm uninstall prefect-server --namespace=hm-prefect
# kubectl delete namespace hm-prefect
echo "=================================================="

echo "# Create hm-prefect profile"
prefect profile create hm-prefect
prefect profile use hm-prefect
prefect config set PREFECT_API_URL=https://prefect.hongbomiao.com/api
echo "=================================================="

echo "# Build hm-prefect-print-platform"
docker build --file=data-orchestration/hm-prefect/workflows/print-platform/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-print-platform:latest .
docker push ghcr.io/hongbo-miao/hm-prefect-print-platform:latest
echo "=================================================="

echo "# Start the workflow"
# calculate
# cd data-orchestration/hm-prefect/workflows/calculate
# poetry run poe set-up
# poetry run poe build -- --params='{"model":{"n":4}}' --work-queue=hm-kubernetes-queue
# poetry run poe run

# greet
# cd data-orchestration/hm-prefect/workflows/greet
# poetry run poe set-up
# poetry run poe build -- --params='{"user":{"first_name":"Hongbo","last_name":"Miao"}}' --work-queue=hm-kubernetes-queue
# poetry run poe run

# print-platform
cd data-orchestration/hm-prefect/workflows/print-platform
poetry run poe set-up
poetry run poe build -- --work-queue=hm-kubernetes-queue
# poetry run poe build -- --work-queue=hm-kubernetes-queue
# poetry run poe build -- --work-queue=hm-local-queue
poetry run poe run

# kubectl delete jobs --all --namespace=hm-prefect
echo "=================================================="

echo "# Start Prefect Agent in local"
poetry run poe prefect-agent-start -- --work-queue=hm-local-queue
echo "=================================================="

echo "# Start Prefect Agents in Kubernetes"
helm upgrade \
  prefect-agent-1 \
  prefect-agent \
  --install \
  --repo=https://prefecthq.github.io/prefect-helm \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/prefect-agent/my-values.yaml
helm upgrade \
  prefect-agent-2 \
  prefect-agent \
  --install \
  --repo=https://prefecthq.github.io/prefect-helm \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/prefect-agent/my-values.yaml
# helm uninstall prefect-agent-1 --namespace=hm-prefect
# helm uninstall prefect-agent-2 --namespace=hm-prefect
echo "=================================================="
