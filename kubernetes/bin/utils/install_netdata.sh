#!/usr/bin/env bash
set -e

echo "# Install Netdata"
# https://github.com/netdata/helmchart
helm upgrade \
  netdata \
  netdata \
  --install \
  --repo=https://netdata.github.io/helmchart \
  --namespace=hm-netdata \
  --create-namespace

# helm uninstall trino --namespace=hm-netdata
# kubectl delete namespace hm-netdata
echo "=================================================="
