#!/usr/bin/env bash
set -e

echo "# Install Confluent Schema Registry"
# https://github.com/bitnami/charts/tree/main/bitnami/schema-registry
helm upgrade \
  confluent-schema-registry \
  oci://registry-1.docker.io/bitnamicharts/schema-registry \
  --install \
  --namespace=hm-confluent-schema-registry \
  --create-namespace \
  --values=kubernetes/manifests/confluent-schema-registry/helm/my-values.yaml
# helm uninstall confluent-schema-registry --namespace=hm-confluent-schema-registry
# kubectl delete namespace hm-confluent-schema-registry
echo "=================================================="
