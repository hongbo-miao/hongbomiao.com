#!/usr/bin/env bash
set -e

echo "# Install UI for Apache Kafka"
# https://docs.kafka-ui.provectus.io/configuration/helm-charts/quick-start
helm upgrade \
  ui-for-apache-kafka \
  kafka-ui \
  --install \
  --repo=https://provectus.github.io/kafka-ui \
  --namespace=hm-ui-for-apache-kafka \
  --create-namespace \
  --values=kubernetes/manifests/kafka-ui/helm/my-values.yaml
# helm uninstall kafka-ui --namespace=hm-ui-for-apache-kafka
# kubectl delete namespace hm-ui-for-apache-kafka
echo "=================================================="
