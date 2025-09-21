#!/usr/bin/env bash
set -e

echo "# Create database hasura_db, opa_db"
psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="create database hasura_db;"
psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="grant all privileges on database hasura_db to admin;"

psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="create database opa_db;"
psql postgresql://postgres:passw0rd@localhost:5432/postgres --command="grant all privileges on database opa_db to admin;"
echo "=================================================="

echo "# Install Hasura"
cp -R hasura-graphql-engine/migrations/ kubernetes/data/hasura/hasura-graphql-engine/migrations/
cp -R hasura-graphql-engine/metadata/ kubernetes/data/hasura/hasura-graphql-engine/metadata/

kubectl apply --filename=kubernetes/manifests/hasura/hm-hasura-namespace.yaml
kubectl apply --filename=kubernetes/manifests/hasura
# kubectl delete --filename=kubernetes/manifests/hasura
# kubectl delete namespace hm-hasura
echo "=================================================="

echo "# Add seed data in opa_db"
kubectl rollout status deployment/hasura-deployment --namespace=hm-hasura
kubectl port-forward service/hasura-service --namespace=hm-hasura 16020:16020 &
cd hasura-graphql-engine
hasura seed apply
pgrep kubectl | xargs kill -9
echo "=================================================="
