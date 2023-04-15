#!/usr/bin/env bash
set -e

echo "# Install Prometheus with Thanos Sidecar, Grafana"
source kubernetes/bin/utils/install_prometheus_thanos_grafana.sh
echo "=================================================="

echo "# Install OpenCost"
helm upgrade \
  opencost \
  opencost \
  --install \
  --repo=https://opencost.github.io/opencost-helm-chart \
  --namespace=hm-opencost \
  --create-namespace \
  --values=kubernetes/manifests/opencost/helm/my-values.yaml
# helm uninstall opencost --namespace=hm-opencost
# kubectl delete namespace hm-opencost
echo "=================================================="
