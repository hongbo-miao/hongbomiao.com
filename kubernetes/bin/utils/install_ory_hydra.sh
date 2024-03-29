#!/usr/bin/env bash
set -e

echo "# Create ory_hydra_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 5432:5432 &
sleep 5
psql postgresql://admin@localhost:5432/postgres --command="create database ory_hydra_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database ory_hydra_db to admin;"
pgrep kubectl | xargs kill -9
echo "=================================================="

echo "# Install ORY Hydra"
kubectl apply --filename=kubernetes/manifests/ory-hydra/hm-ory-hydra-namespace.yaml
helm upgrade \
  ory-hydra \
  hydra \
  --install \
  --repo=https://k8s.ory.sh/helm/charts \
  --namespace=hm-ory-hydra \
  --create-namespace \
  --values=kubernetes/manifests/ory-hydra/helm/my-values.yaml
# helm uninstall ory-hydra --namespace=hm-ory-hydra
# kubectl delete namespace hm-ory-hydra
echo "=================================================="
