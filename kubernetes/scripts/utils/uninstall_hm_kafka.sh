#!/usr/bin/env bash
set -e

echo "# Uninstall Kafka"
# kubectl delete --filename=kubernetes/manifests/hm-kafka/hm-kafka/kafka-persistent.yaml --namespace=hm-kafka
kubectl delete --filename=kubernetes/manifests/hm-kafka/hm-kafka/kafka-persistent-single.yaml --namespace=hm-kafka
kubectl delete --filename="https://strimzi.io/install/latest?namespace=hm-kafka"
kubectl delete namespace hm-kafka
echo "=================================================="

echo "# Delete Kafka truststore"
KAFKA_DATA_PATH="kubernetes/data/hm-kafka/hm-kafka"
rm -f "${KAFKA_DATA_PATH}/ca.crt"
rm -f "${KAFKA_DATA_PATH}/kafka-truststore.jks"
echo "=================================================="
