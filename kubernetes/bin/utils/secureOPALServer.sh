#!/usr/bin/env bash

set -e


echo "# Generate OPAL SSH key"
rm -f kubernetes/data/opal-server/id_rsa
rm -f kubernetes/data/opal-server/id_rsa.pub
ssh-keygen -t rsa -b 4096 -m pem -f "$PWD/kubernetes/data/opal-server/id_rsa" -N ""
echo "=================================================="

echo "# Create OPAL server secret"
OPAL_AUTH_MASTER_TOKEN=IWjW0bYcTIfm6Y5JNjp4DdgopC6rYSxT4yrPbtLiTU0
kubectl create secret generic hm-opal-server-secret \
  --namespace=hm-opa \
  --from-file=id_rsa=kubernetes/data/opal-server/id_rsa \
  --from-file=id_rsa.pub=kubernetes/data/opal-server/id_rsa.pub \
  --from-literal=opal_auth_master_token="${OPAL_AUTH_MASTER_TOKEN}"
echo "=================================================="

echo "# Check OPAL server"
kubectl rollout status deployment/opal-server-deployment --namespace=hm-opa
echo "=================================================="

echo "# Create OPAL client secret"
kubectl port-forward service/opal-server-service --namespace=hm-opa 7002:7002 &
OPAL_CLIENT_TOKEN=$(curl --location --request POST "http://localhost:7002/token" \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer ${OPAL_AUTH_MASTER_TOKEN}" \
  --data-raw '{
    "type": "client"
  }' | \
  jq '.token' --raw-output)
kubectl create secret generic hm-opal-client-secret \
  --namespace=hm \
  --from-literal=opal_client_token="${OPAL_CLIENT_TOKEN}"
pgrep kubectl | xargs kill -9
echo "=================================================="
