#!/usr/bin/env bash
set -e

echo "# Install Grafana"
helm upgrade \
  grafana \
  grafana \
  --install \
  --repo=https://grafana.github.io/helm-charts \
  --namespace=hm-grafana \
  --create-namespace \
  --values=kubernetes/manifests/grafana/helm/my-values.yaml
# helm uninstall grafana --namespace=hm-grafana
# kubectl delete namespace hm-grafana
echo "=================================================="

echo "# Install Loki"
helm upgrade \
  loki \
  loki \
  --install \
  --repo=https://grafana.github.io/helm-charts \
  --namespace=hm-loki \
  --create-namespace \
  --values=kubernetes/manifests/loki/helm/my-values.yaml
# helm uninstall loki --namespace=hm-loki
# kubectl delete namespace hm-loki
echo "=================================================="
