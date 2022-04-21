#!/usr/bin/env bash
set -e

echo "# Create ory_hydra_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072 &
sleep 5
psql --host=localhost --port=40072 --dbname=postgres --username=admin --command="create database ory_hydra_db;"
psql --host=localhost --port=40072 --dbname=postgres --username=admin --command="grant all privileges on database ory_hydra_db to admin;"
pgrep kubectl | xargs kill -9
echo "=================================================="

echo "# Install ORY Hydra"
kubectl apply --filename=kubernetes/manifests/ory-hydra/hm-ory-hydra-namespace.yaml
helm repo add ory https://k8s.ory.sh/helm/charts
helm repo update
helm install \
  ory-hydra \
  ory/hydra \
  --namespace=hm-ory-hydra \
  --values=ory-hydra/ory-hydra.yaml

# Upgrade:
# helm upgrade \
#   ory-hydra \
#   ory/hydra \
#   --namespace=hm-ory-hydra \
#   --values=ory-hydra/ory-hydra.yaml

# Delete:
# helm uninstall ory-hydra --namespace=hm-ory-hydra
echo "=================================================="
