#!/usr/bin/env bash
set -e

echo "# Install Qdrant"
# https://github.com/qdrant/qdrant-helm
helm upgrade \
  qdrant \
  qdrant \
  --install \
  --repo=https://qdrant.github.io/qdrant-helm \
  --namespace=hm-qdrant \
  --create-namespace
# helm uninstall qdrant --namespace=hm-qdrant
# kubectl delete namespace hm-qdrant
echo "=================================================="
