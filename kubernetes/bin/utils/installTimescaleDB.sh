#!/usr/bin/env bash
set -e

echo "# Install TimescaleDB"
# https://docs.timescale.com/install/latest/installation-kubernetes/#install-timescaledb-on-kubernetes
# https://github.com/FairwindsOps/charts/blob/master/stable/goldilocks/values.yaml
helm repo add timescale https://charts.timescale.com
helm repo update

kubectl apply --filename=kubernetes/manifests/timescale/hm-timescale-namespace.yaml
helm install \
  timescale \
  timescale/timescaledb-single \
  --namespace=hm-timescale \
  --values=kubernetes/manifests/timescale/helm/my-values.yaml

# Delete:
# helm uninstall timescale --namespace=hm-timescale
echo "=================================================="
