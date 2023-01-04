#!/usr/bin/env bash
set -e

CLUSTERS=("$@")

# Install Ingress with patch
for cluster in "${CLUSTERS[@]}"; do
  echo "# Install Ingress on: k3d-${cluster}"
  INGRESS_VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl apply \
    --context="k3d-${cluster}" \
    --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_VERSION}/deploy/static/provider/cloud/deploy.yaml"
  # Local: kubectl apply --context="k3d-${cluster}" --filename=kubernetes/manifests-raw/ingress/ingress-nginx.yaml
  echo "=================================================="
done
