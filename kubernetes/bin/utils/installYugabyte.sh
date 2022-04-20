#!/usr/bin/env bash
set -e

echo "# Deploy Yugabyte"
YUGABYTE_PATH="kubernetes/data/yugabyte"
YUGABYTE_REPLICA=3
for ((i = 0; i < YUGABYTE_REPLICA; i++)); do
  rm -rf "${YUGABYTE_PATH}/master-${i}"
  rm -rf "${YUGABYTE_PATH}/tserver-${i}"
  mkdir "${YUGABYTE_PATH}/master-${i}"
  mkdir "${YUGABYTE_PATH}/tserver-${i}"
done
kubectl apply --filename=kubernetes/manifests-raw/yugabyte/crds/yugabyte.com_ybclusters_crd.yaml
kubectl apply --filename=kubernetes/manifests-raw/yugabyte/operator.yaml
kubectl apply --filename=kubernetes/manifests/yugabyte
sleep 60
echo "=================================================="

echo "# Check Yugabyte"
kubectl rollout status deployment/yugabyte-operator --namespace=yb-operator
kubectl wait --for=condition=ready pod --selector=app=yb-master --namespace=yb-operator
kubectl wait --for=condition=ready pod --selector=app=yb-tserver --namespace=yb-operator
echo "=================================================="

echo "# Create opa_db in Yugabyte"
kubectl port-forward service/yb-tservers --namespace=yb-operator 5433:5433 &
sleep 5
psql --host=localhost --port=5433 --dbname=yugabyte --username=yugabyte --command="create database opa_db;"
psql --host=localhost --port=5433 --dbname=yugabyte --username=yugabyte --command="create role admin with login password 'passw0rd';"
psql --host=localhost --port=5433 --dbname=yugabyte --username=yugabyte --command="grant all privileges on database opa_db to admin;"
psql --host=localhost --port=5433 --dbname=opa_db --username=yugabyte --command="create extension if not exists pgcrypto;"
echo "=================================================="

echo "# Initialize OPA Data in Yugabyte"
POSTGRESQL_URL="postgres://admin:passw0rd@localhost:5433/opa_db?sslmode=disable&search_path=public"
migrate -database "${POSTGRESQL_URL}" -path kubernetes/data/postgres/opa_db/migrations up
pgrep kubectl | xargs kill -9
echo "=================================================="
