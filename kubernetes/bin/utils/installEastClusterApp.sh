#!/usr/bin/env bash
set -e

echo "# Install the app"
kubectl apply --filename=kubernetes/manifests/east/hm-namespace.yaml
kubectl apply --filename=kubernetes/manifests/east
# Delete: kubectl delete --filename=kubernetes/manifests/east
echo "=================================================="
sleep 30
