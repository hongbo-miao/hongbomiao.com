#!/usr/bin/env bash

set -e


kubectl create secret generic hm-opa-db-credentials \
  --from-file=kubernetes/config/debezium/debezium-opa-db-credentials.properties \
  --namespace=kafka

kubectl apply --filename=kubernetes/config/debezium
