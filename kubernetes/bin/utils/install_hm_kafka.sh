#!/usr/bin/env bash
set -e

# Kafka
# https://strimzi.io/quickstarts
echo "# Install Strimzi"
kubectl create namespace hm-kafka
# kubectl delete namespace hm-kafka
kubectl apply --filename="https://strimzi.io/install/latest?namespace=hm-kafka" --namespace=hm-kafka
# kubectl delete --filename="https://strimzi.io/install/latest?namespace=kafka" --namespace=hm-kafka
echo "=================================================="

echo "# Install Kafka"
kubectl apply --filename=https://strimzi.io/examples/latest/kafka/kafka-persistent-single.yaml --namespace=hm-kafka
# kubectl delete --filename=https://strimzi.io/examples/latest/kafka/kafka-persistent-single.yaml --namespace=hm-kafka
echo "=================================================="

echo "# Check Kafka"
kubectl wait kafka/my-cluster --for=condition=Ready --timeout=300s --namespace=hm-kafka
# kubectl run kafka-producer --stdin --tty --namespace=hm-kafka --image=quay.io/strimzi/kafka:0.26.0-kafka-3.0.0 --rm=true --restart=Never -- bin/kafka-console-producer.sh --broker-list=my-cluster-kafka-bootstrap:9092 --topic=my-topic
# kubectl run kafka-consumer --stdin --tty --namespace=hm-kafka --image=quay.io/strimzi/kafka:0.26.0-kafka-3.0.0 --rm=true --restart=Never -- bin/kafka-console-consumer.sh --bootstrap-server=my-cluster-kafka-bootstrap:9092 --topic=my-topic --from-beginning
echo "=================================================="
