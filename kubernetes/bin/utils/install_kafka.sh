#!/usr/bin/env bash
set -e

# Kafka
# https://strimzi.io/quickstarts
echo "# Install Strimzi"
kubectl create namespace kafka
kubectl apply --filename="https://strimzi.io/install/latest?namespace=kafka" --namespace=kafka
# kubectl delete --filename="https://strimzi.io/install/latest?namespace=kafka" --namespace=kafka
echo "=================================================="

echo "# Install Kafka"
kubectl apply --filename=https://strimzi.io/examples/latest/kafka/kafka-persistent-single.yaml --namespace=kafka
# kubectl delete --filename=https://strimzi.io/examples/latest/kafka/kafka-persistent-single.yaml --namespace=kafka
echo "=================================================="

echo "# Check Kafka"
kubectl wait kafka/my-cluster --for=condition=Ready --timeout=300s --namespace=kafka
# kubectl run kafka-producer --stdin --tty --namespace=kafka --image=quay.io/strimzi/kafka:0.26.0-kafka-3.0.0 --rm=true --restart=Never -- bin/kafka-console-producer.sh --broker-list=my-cluster-kafka-bootstrap:9092 --topic=my-topic
# kubectl run kafka-consumer --stdin --tty --namespace=kafka --image=quay.io/strimzi/kafka:0.26.0-kafka-3.0.0 --rm=true --restart=Never -- bin/kafka-console-consumer.sh --bootstrap-server=my-cluster-kafka-bootstrap:9092 --topic=my-topic --from-beginning
echo "=================================================="
