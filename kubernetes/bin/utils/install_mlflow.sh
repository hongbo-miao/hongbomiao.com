#!/usr/bin/env bash
set -e

echo "# Create hm_mlflow_db in Postgres"
kubectl port-forward service/postgres-service --namespace=hm-postgres 5432:5432 &
sleep 5
psql postgresql://admin@localhost:5432/postgres --command="create database hm_mlflow_db;"
psql postgresql://admin@localhost:5432/postgres --command="grant all privileges on database hm_mlflow_db to admin;"
pgrep kubectl | xargs kill -9
echo "=================================================="

echo "# Install MLflow"
# https://github.com/community-charts/helm-charts/tree/main/charts/mlflow
helm upgrade \
  mlflow \
  mlflow \
  --install \
  --repo=https://community-charts.github.io/helm-charts \
  --namespace=hm-mlflow \
  --create-namespace \
  --values=kubernetes/manifests/mlflow/helm/my-values.yaml
# helm uninstall mlflow --namespace=hm-mlflow
# kubectl delete namespace hm-mlflow
echo "=================================================="
