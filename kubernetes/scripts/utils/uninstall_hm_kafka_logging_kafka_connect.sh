#!/usr/bin/env bash
set -e

echo "# Delete elasticsearch-keystore.jks"
KAFKACONNECT_DATA_PATH="kubernetes/data/hm-kafka/logging-kafka-connect"
rm -f "${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.jks"
rm -f "${KAFKACONNECT_DATA_PATH}/tls.key"
rm -f "${KAFKACONNECT_DATA_PATH}/tls.crt"
rm -f "${KAFKACONNECT_DATA_PATH}/ca.crt"
rm -f "${KAFKACONNECT_DATA_PATH}/elasticsearch-keystore.p12"
echo "=================================================="

echo "# Delete secret hm-elasticsearch-keystore"
kubectl delete secret hm-elasticsearch-keystore --namespace=hm-kafka
echo "=================================================="

echo "# Delete secret hm-elasticsearch-credentials"
kubectl delete secret hm-elasticsearch-credentials --namespace=hm-kafka
echo "=================================================="

echo "# Delete hm-kafka-logging-kafka-connect"
kubectl delete --filename=kubernetes/manifests/hm-kafka/logging-kafka-connect
echo "=================================================="
