#!/usr/bin/env bash
set -e

echo "# Install Sloop"
# https://github.com/salesforce/sloop/blob/master/helm/sloop/README.md
helm install \
  sloop \
  submodules/sloop/helm/sloop \
  --namespace=hm-sloop \
  --create-namespace \
  --values=kubernetes/manifests/sloop/helm/my-values.yaml
# helm uninstall sloop --namespace=hm-sloop
# kubectl delete namespace hm-sloop
echo "=================================================="
