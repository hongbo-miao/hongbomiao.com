#!/usr/bin/env bash
set -e

OPAL_SERVER_DATA_PATH="kubernetes/data/opal-server"

echo "# Clean OPAL SSH key"
rm -f "${OPAL_SERVER_DATA_PATH}/opal_auth_private_key.pem"
rm -f "${OPAL_SERVER_DATA_PATH}/opal_auth_public_key.pem"
rm -f kubernetes/data/config-server/opal_auth_public_key.pem
echo "=================================================="

echo "# Generate OPAL SSH key"
OPAL_AUTH_PRIVATE_KEY_PASSPHRASE="ds6l3qYYx9UsYcgshmlbsMJTXs1lVH9ndf13Xp1xNKxbqjFdxFvdkJxpm0DfjAhh"
ssh-keygen -t rsa -b 4096 -m pem -f "${OPAL_SERVER_DATA_PATH}/opal_auth_private_key.pem" -N "${OPAL_AUTH_PRIVATE_KEY_PASSPHRASE}"
rm -f "${OPAL_SERVER_DATA_PATH}/opal_auth_private_key.pem.pub"
ssh-keygen -e -m pkcs8 -f "${OPAL_SERVER_DATA_PATH}/opal_auth_private_key.pem" -P "${OPAL_AUTH_PRIVATE_KEY_PASSPHRASE}" > "${OPAL_SERVER_DATA_PATH}/opal_auth_public_key.pem"
cp "${OPAL_SERVER_DATA_PATH}/opal_auth_public_key.pem" api-go/config/config_server/opal_auth_public_key.pem
cp "${OPAL_SERVER_DATA_PATH}/opal_auth_public_key.pem" kubernetes/data/config-server/opal_auth_public_key.pem
echo "=================================================="

echo "# Create OPAL server secret"
OPAL_AUTH_MASTER_TOKEN=IWjW0bYcTIfm6Y5JNjp4DdgopC6rYSxT4yrPbtLiTU0
kubectl create secret generic hm-opal-server-secret \
  --namespace=hm-opa \
  --from-file="opal_auth_private_key=${OPAL_SERVER_DATA_PATH}/opal_auth_private_key.pem" \
  --from-literal="opal_auth_private_key_passphrase=${OPAL_AUTH_PRIVATE_KEY_PASSPHRASE}" \
  --from-file="opal_auth_public_key=${OPAL_SERVER_DATA_PATH}/opal_auth_public_key.pem" \
  --from-literal="opal_auth_master_token=${OPAL_AUTH_MASTER_TOKEN}"
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
    "type": "datasource",
    "claims": {
        "client_id": "hm-opal-client"
    }
  }' | \
  jq '.token' --raw-output)
kubectl create secret generic hm-opal-client-secret \
  --namespace=hm \
  --from-literal="opal_client_token=${OPAL_CLIENT_TOKEN}" \
  --from-literal="opal_default_update_callbacks={\"callbacks\":[(\"http://opal-server-service.hm-opa:7002/data/callback_report\",{\"headers\":{\"Authorization\":\"Bearer ${OPAL_CLIENT_TOKEN}\"}})]}"
pgrep kubectl | xargs kill -9
echo "=================================================="
