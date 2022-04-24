#!/usr/bin/env bash
set -e

echo "# Install Polaris"
# https://polaris.docs.fairwinds.com/dashboard/#using-the-dashboard
kubectl apply --filename=kubernetes/manifests/polaris/hm-polaris-namespace.yaml
helm repo add fairwinds-stable https://charts.fairwinds.com/stable
helm install \
  polaris \
  fairwinds-stable/polaris \
  --namespace=hm-polaris \
  --values=kubernetes/manifests/polaris/helm/my-values.yaml

# Delete:
# helm uninstall polaris --namespace=hm-polaris
echo "=================================================="
