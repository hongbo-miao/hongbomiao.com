#!/usr/bin/env bash
set -e

echo "# Install Prometheus with Thanos Sidecar, Grafana"
# https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack
helm upgrade \
  monitoring \
  kube-prometheus-stack \
  --install \
  --repo=https://prometheus-community.github.io/helm-charts \
  --namespace=hm-monitoring \
  --create-namespace \
  --values=kubernetes/manifests/prometheus/helm/my-values.yaml
# helm uninstall monitoring --namespace=hm-monitoring
# kubectl delete namespace hm-monitoring

# Grafana
# Username: admin
# Password: passw0rd
echo "=================================================="
