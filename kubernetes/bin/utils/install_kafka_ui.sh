#!/usr/bin/env bash
set -e

echo "# Install Kafka UI"
# https://github.com/eshepelyuk/apicurio-registry-helm
helm upgrade \
  kafka-ui \
  kafka-ui \
  --install \
  --repo=https://provectus.github.io/kafka-ui \
  --namespace=hm-kafka-ui \
  --create-namespace \
  --values=kubernetes/manifests/kafka-ui/helm/my-values.yaml
# helm uninstall kafka-ui --namespace=hm-kafka-ui
# kubectl delete namespace hm-kafka-ui
echo "=================================================="
