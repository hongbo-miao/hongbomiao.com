#!/usr/bin/env bash
set -e

echo "# Install Sloop"
# https://github.com/salesforce/sloop/blob/master/helm/sloop/README.md
kubectl apply --filename=kubernetes/manifests/sloop/hm-sloop-namespace.yaml
helm install \
  sloop \
  submodules/sloop/helm/sloop \
  --namespace=hm-sloop \
  --values=kubernetes/manifests/sloop/helm/my-values.yaml

# Delete:
# helm uninstall sloop --namespace=hm-sloop
echo "=================================================="
