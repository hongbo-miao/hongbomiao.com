#!/usr/bin/env bash
set -e

echo "# Install Apicurio Registry"
# https://github.com/eshepelyuk/apicurio-registry-helm
helm upgrade \
  apicurio-registry \
  oci://ghcr.io/eshepelyuk/helm/apicurio-registry \
  --install \
  --namespace=hm-apicurio-registry \
  --create-namespace \
  --values=kubernetes/manifests/apicurio-registry/helm/my-values.yaml
# helm uninstall apicurio-registry --namespace=hm-apicurio-registry
# kubectl delete namespace hm-apicurio-registry
echo "=================================================="
