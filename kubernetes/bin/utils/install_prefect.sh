#!/usr/bin/env bash
set -e

echo "# Install cloudflared"
source kubernetes/bin/utils/install_cloudflared.sh
echo "=================================================="

echo "# Install Postgres"
source kubernetes/bin/utils/install_postgres.sh
psql postgresql://admin@localhost:5432/postgres --command="create database hm_prefect_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database hm_prefect_db to admin;"
echo "=================================================="

echo "# Install Prefect Server"
helm repo add prefect https://prefecthq.github.io/prefect-helm
helm repo update prefect
kubectl create secret generic hm-prefect-postgres-secret \
  --namespace=hm-prefect \
  --from-literal="connection-string=postgresql+asyncpg://admin:passw0rd@postgres-service.hm-postgres.svc:5432/hm_prefect_db"
# kubectl delete secret hm-prefect-postgres-secret --namespace=hm-prefect

helm install \
  prefect-server \
  prefect/prefect-server \
  --namespace=hm-prefect \
  --create-namespace \
  --values=kubernetes/manifests/prefect/helm/prefect-server/my-values.yaml
# helm uninstall prefect-server --namespace=hm-prefect
echo "=================================================="

echo "# Create hm-kubernetes profile"
prefect profile create hm-kubernetes
prefect profile use hm-kubernetes
prefect config set PREFECT_API_URL=https://prefect.hongbomiao.com/api
echo "=================================================="

echo "# Build hm-prefect-print-platform"
docker build --file=hm-prefect/workflows/print-platform/Dockerfile --tag=ghcr.io/hongbo-miao/hm-prefect-print-platform:latest .
docker push ghcr.io/hongbo-miao/hm-prefect-print-platform:latest
echo "=================================================="

cd hm-prefect/workflows/print-platform
# cd hm-prefect/workflows/collect-data

echo "# Start the workflow"
poetry run poe add-kubernetes-job-block
poetry run poe build -- --work-queue=hm-local-queue
# poetry run poe build -- --params=$(cat params.json | jq -c .) --work-queue=hm-local-queue
# poetry run poe build -- --work-queue=hm-kubernetes-queue
poetry run poe run
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
# helm uninstall prefect-agent --namespace=hm-prefect
echo "=================================================="
