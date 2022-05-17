#!/usr/bin/env bash
set -e

echo "# Install Temporal"
# https://github.com/temporalio/helm-charts
cd submodules/temporal/helm-charts
helm dependencies update
cd ../../..
helm install \
  temporal \
  submodules/temporal/helm-charts \
  --namespace=hm-temporal \
  --create-namespace \
  --values=kubernetes/manifests/temporal/helm/my-values.yaml \
  --timeout=15m

# Delete:
# helm uninstall temporal --namespace=hm-temporal
# kubectl delete job temporal-schema-setup --namespace=hm-temporal
echo "=================================================="
