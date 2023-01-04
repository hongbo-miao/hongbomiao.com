#!/usr/bin/env bash
set -e

echo "# Install Hasura"
cp -r hasura-graphql-engine/migrations/ kubernetes/data/hasura/hasura-graphql-engine/migrations/
cp -r hasura-graphql-engine/metadata/ kubernetes/data/hasura/hasura-graphql-engine/metadata/

kubectl apply --filename=kubernetes/manifests/hasura/hm-hasura-namespace.yaml
kubectl apply --filename=kubernetes/manifests/hasura
# kubectl delete --filename=kubernetes/manifests/hasura
echo "=================================================="

echo "# Add seed data in opa_db"
kubectl rollout status deployment/hasura-deployment --namespace=hm-hasura
kubectl port-forward service/hasura-service --namespace=hm-hasura 16020:16020 &
cd hasura-graphql-engine
hasura seed apply
pgrep kubectl | xargs kill -9
echo "=================================================="
