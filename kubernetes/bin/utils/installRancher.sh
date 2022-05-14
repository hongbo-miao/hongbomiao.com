#!/usr/bin/env bash
set -e

# https://rancher.com/docs/rancher/v2.6/en/installation/install-rancher-on-k8s/

echo "# Install Cert Manager"
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install \
  cert-manager \
  jetstack/cert-manager \
  --namespace=hm-cert-manager \
  --create-namespace \
  --version=v1.8.0 \
  --set=installCRDs=true

# Delete:
# helm uninstall cert-manager --namespace=hm-cert-manager
echo "=================================================="

echo "# Install Rancher"
# https://github.com/rancher/rancher/blob/release/v2.6/chart/values.yaml
helm repo add rancher-latest https://releases.rancher.com/server-charts/latest
helm repo update
helm install \
  rancher \
  rancher-latest/rancher \
  --namespace=cattle-system \
  --create-namespace \
  --values=kubernetes/manifests/rancher/helm/my-values.yaml

# Delete:
# helm uninstall rancher --namespace=cattle-system

# kubectl port-forward service/rancher --namespace=cattle-system 46271:443
# https://localhost:46271
# password: passw0rd
echo "=================================================="
