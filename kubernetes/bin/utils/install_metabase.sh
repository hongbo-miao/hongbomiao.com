#!/usr/bin/env bash
set -e

echo "# Install Postgres"
helm repo add pmint93 https://pmint93.github.io/helm-charts
helm repo update pmint93
helm install \
  metabase \
  pmint93/metabase \
  --namespace=hm-metabase \
  --create-namespace
echo "=================================================="
