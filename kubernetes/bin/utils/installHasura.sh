#!/usr/bin/env bash
set -e

echo "# Install Hasura"
cp -r hasura-graphql-engine/migrations/ kubernetes/data/hasura/migrations/
cp -r hasura-graphql-engine/metadata/ kubernetes/data/hasura/metadata/

kubectl apply --filename=kubernetes/manifests/hasura/hm-hasura-namespace.yaml
kubectl apply --filename=kubernetes/manifests/hasura
echo "=================================================="

echo "# Add seed data in opa_db"
kubectl port-forward service/postgres-service --namespace=hm-postgres 40072:40072 &
sleep 5
cd hasura-graphql-engine
hasura seed apply
pgrep kubectl | xargs kill -9
echo "=================================================="
