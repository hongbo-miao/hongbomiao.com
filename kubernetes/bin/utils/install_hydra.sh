#!/usr/bin/env bash
set -e

echo "# Install Hydrea"
kubectl apply --filename=kubernetes/manifests/hydra
# kubectl delete --filename=kubernetes/manifests/hydra
echo "=================================================="
