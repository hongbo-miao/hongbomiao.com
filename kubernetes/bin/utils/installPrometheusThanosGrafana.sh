#!/usr/bin/env bash
set -e

echo "# Install Prometheus with Thanos Sidecar, Grafana"
# https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
# helm search repo prometheus-community
helm install \
  monitoring \
  prometheus-community/kube-prometheus-stack \
  --namespace=hm-monitoring \
  --create-namespace \
  --values=kubernetes/manifests/prometheus/helm/my-values.yaml

# Upgrade:
# helm upgrade \
#   monitoring \
#   prometheus-community/kube-prometheus-stack \
#   --namespace=hm-monitoring \
#   --values=kubernetes/manifests/prometheus/helm/my-values.yaml

# Delete:
# helm uninstall monitoring --namespace=hm-monitoring

# Grafana
# Username: admin
# Password: passw0rd
echo "=================================================="
