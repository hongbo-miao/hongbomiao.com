#!/usr/bin/env bash

set -e


echo "# Install Debezium"
kubectl create secret generic hm-opa-db-credentials \
  --from-file=kubernetes/config/debezium/opa-db-credentials.properties \
  --namespace=kafka

kubectl apply --filename=kubernetes/config/debezium
echo "=================================================="
