#!/usr/bin/env bash
set -e


echo "# Create secret hm-opa-db-credentials"
kubectl create secret generic hm-opa-db-credentials \
  --from-file=kubernetes/manifests/kafka/postgres-kafkaconnector/opa-db-credentials.properties \
  --namespace=kafka
echo "=================================================="

echo "# Clean elasticsearch-keystore.jks"
KAFKACONNECT_DATA_PATH="kubernetes/data/kafkaconnect"
rm -f "${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.jks"
echo "=================================================="

echo "# Create elasticsearch-keystore.jks"
KEYSTORE_P12_PASSWORD="SFLzyT8DPkGGjDtn"
KEYSTORE_JSK_PASSWORD="MPx57vkACsRWKVap"

kubectl get secret hm-elasticsearch-es-http-certs-public \
  --namespace=elastic \
  --output=go-template='{{index .data "ca.crt" | base64decode }}' \
  > "${KAFKACONNECT_DATA_PATH}/ca.crt"

kubectl get secret hm-elasticsearch-es-http-certs-public \
  --namespace=elastic \
  --output=go-template='{{index .data "tls.crt" | base64decode }}' \
  > "${KAFKACONNECT_DATA_PATH}/tls.crt"

kubectl get secret hm-elasticsearch-es-http-certs-internal \
  --namespace=elastic \
  --output=go-template='{{index .data "tls.key" | base64decode }}' \
  > "${KAFKACONNECT_DATA_PATH}/tls.key"

openssl pkcs12 -export \
  -in "${KAFKACONNECT_DATA_PATH}/tls.crt" \
  -inkey "${KAFKACONNECT_DATA_PATH}/tls.key" \
  -CAfile "${KAFKACONNECT_DATA_PATH}/ca.crt" \
  -caname root \
  -out "${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.p12" \
  -password "pass:${KEYSTORE_P12_PASSWORD}" \
  -name hm-elasticsearch-keystore

keytool -importkeystore \
  -srckeystore  "${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.p12" \
  -srcstoretype PKCS12 \
  -srcstorepass ${KEYSTORE_P12_PASSWORD} \
  -destkeystore  "${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.jks" \
  -destkeypass ${KEYSTORE_JSK_PASSWORD} \
  -deststorepass ${KEYSTORE_JSK_PASSWORD} \
  -alias hm-elasticsearch-keystore

rm -f "${KAFKACONNECT_DATA_PATH}/tls.key"
rm -f "${KAFKACONNECT_DATA_PATH}/tls.crt"
rm -f "${KAFKACONNECT_DATA_PATH}/ca.crt"
rm -f "${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.p12"
echo "=================================================="

echo "# Create secret hm-elasticsearch-keystore"
kubectl create secret generic hm-elasticsearch-keystore \
  --from-file="${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.jks" \
  --namespace=kafka
echo "=================================================="

echo "# Create secret hm-elasticsearch-credentials"
kubectl create secret generic hm-elasticsearch-credentials \
  --from-file=kubernetes/manifests/kafka/elasticsearch-sink-kafkaconnector/elasticsearch-credentials.properties \
  --namespace=kafka
echo "=================================================="

echo "# Install Debezium"
kubectl apply --filename=kubernetes/manifests/kafka
echo "=================================================="
