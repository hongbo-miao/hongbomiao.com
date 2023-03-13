#!/usr/bin/env bash
set -e

echo "# Install Prometheus with Thanos Sidecar, Grafana"
source kubernetes/bin/utils/install_prometheus_thanos_grafana.sh
echo "=================================================="

echo "# Install OpenCost"
helm repo add opencost https://opencost.github.io/opencost-helm-chart
helm install \
  opencost \
  opencost/opencost \
  --namespace=hm-opencost \
  --create-namespace \
  --values=kubernetes/manifests/opencost/helm/my-values.yaml

# Delete:
# helm uninstall opencost --namespace=hm-opencost
echo "=================================================="
