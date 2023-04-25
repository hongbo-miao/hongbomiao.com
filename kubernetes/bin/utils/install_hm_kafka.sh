#!/usr/bin/env bash
set -e

echo "# Install Ingress"
# Note `enable-ssl-passthrough: true` in kubernetes/manifests/ingress-nginx/helm/my-values.yaml
# is required to make Kakfa accessible for clients running outside of Kubernetes
helm upgrade \
  ingress-nginx \
  ingress-nginx \
  --install \
  --repo=https://kubernetes.github.io/ingress-nginx \
  --namespace=ingress-nginx \
  --create-namespace \
  --values=kubernetes/manifests/ingress-nginx/helm/my-values.yaml
# helm uninstall ingress-nginx --namespace=ingress-nginx
# kubectl delete namespace ingress-nginx
echo "=================================================="

echo "# Get IP address by EXTERNAL-IP"
# kubectl get nodes
# kubectl get node lima-rancher-desktop --output=wide
echo "=================================================="

echo "# Update IP address in the Kafka bootstrap and brokers"
# Update IP address in kubernetes/manifests/hm-kafka/kafka-persistent.yaml
# Update IP address in below "Produce messages to Kafka" section
echo "=================================================="

echo "# Install Kafka"
# https://strimzi.io/quickstarts
kubectl create namespace hm-kafka
kubectl apply --filename="https://strimzi.io/install/latest?namespace=hm-kafka"
kubectl apply --filename=kubernetes/manifests/hm-kafka/hm-kafka/kafka-persistent-single.yaml --namespace=hm-kafka
# kubectl apply --filename=kubernetes/manifests/hm-kafka/hm-kafka/kafka-persistent.yaml --namespace=hm-kafka

# kubectl delete --filename=kubernetes/manifests/hm-kafka/hm-kafka/kafka-persistent.yaml --namespace=hm-kafka
# kubectl delete --filename=kubernetes/manifests/hm-kafka/hm-kafka/kafka-persistent-single.yaml --namespace=hm-kafka
# kubectl delete --filename="https://strimzi.io/install/latest?namespace=hm-kafka"
# kubectl delete namespace hm-kafka
echo "=================================================="

echo "# Check Kafka"
kubectl wait kafka/hm-kafka --for=condition=Ready --timeout=300s --namespace=hm-kafka
echo "=================================================="

echo "# Get Kafka truststore"
KAFKA_DATA_PATH="kubernetes/data/hm-kafka/hm-kafka"
KAFKA_TRUSTSTORE_PASSWORD="m1Uaf4Crxzoo2Zxp"

rm -f "${KAFKA_DATA_PATH}/ca.crt"
rm -f "${KAFKA_DATA_PATH}/kafka-truststore.jks"

kubectl get secret hm-kafka-cluster-ca-cert \
  --namespace=hm-kafka \
  --output=jsonpath="{.data.ca\.crt}" \
  | base64 -d \
  > "${KAFKA_DATA_PATH}/ca.crt"
keytool \
  -importcert \
  -trustcacerts \
  -file "${KAFKA_DATA_PATH}/ca.crt" \
  -keystore "${KAFKA_DATA_PATH}/kafka-truststore.jks" \
  -storepass "${KAFKA_TRUSTSTORE_PASSWORD}" \
  -alias hm-kafka-truststore \
  -noprompt
echo "=================================================="

echo "# Produce messages to Kafka"
kafka-console-producer \
  --broker-list=kafka-bootstrap.10.10.8.135.nip.io:443 \
  --producer-property=security.protocol=SSL \
  --producer-property=ssl.truststore.location="${KAFKA_DATA_PATH}/kafka-truststore.jks" \
  --producer-property=ssl.truststore.password="${KAFKA_TRUSTSTORE_PASSWORD}" \
  --topic=my-topic
echo "=================================================="

echo "# Consume messages from Kafka"
kubectl exec hm-kafka-kafka-0 --namespace=hm-kafka --container=kafka --stdin --tty -- \
  bin/kafka-console-consumer.sh \
    --bootstrap-server=localhost:9092 \
    --topic=my-topic \
    --property=print.key=true \
    --from-beginning
echo "=================================================="
