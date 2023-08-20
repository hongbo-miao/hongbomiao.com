#!/usr/bin/env bash
set -e

echo "# Install Postgres Operator"
# https://github.com/zalando/postgres-operator/blob/master/docs/quickstart.md#deployment-options
helm upgrade \
  postgres-operator \
  postgres-operator \
  --install \
  --repo=https://opensource.zalando.com/postgres-operator/charts/postgres-operator \
  --namespace=hm-postgres-operator \
  --create-namespace

helm upgrade \
  postgres-operator-ui \
  postgres-operator-ui \
  --install \
  --repo=https://opensource.zalando.com/postgres-operator/charts/postgres-operator-ui \
  --namespace=hm-postgres-operator \
  --create-namespace
echo "=================================================="

echo "# Deploy a Postgres cluster"
kubectl create namespace hm-postgres
kubectl apply --filename=kubernetes/manifests/postgres-operator/postgres
# kubectl delete postgresql hm-postgres-cluster --namespace=hm-postgres
# kubectl delete namespace hm-postgres
echo "=================================================="
