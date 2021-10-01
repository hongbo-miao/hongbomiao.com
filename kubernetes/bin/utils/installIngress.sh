#!/usr/bin/env bash

set -e


# Install Ingress with patch
for cluster in west east; do
  echo "# Install Ingress on: k3d-${cluster}"
  INGRESS_VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl apply \
    --context="k3d-${cluster}" \
    --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_VERSION}/deploy/static/provider/cloud/deploy.yaml"
  # Local: kubectl apply --context="k3d-${cluster}" --filename=kubernetes/manifests/ingress/ingress-nginx.yaml
  echo "=================================================="
done
