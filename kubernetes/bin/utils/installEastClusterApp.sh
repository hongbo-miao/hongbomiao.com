#!/usr/bin/env bash
set -e

echo "# Install the app"
kubectl apply --filename=kubernetes/config/east/hm-namespace.yaml
kubectl apply --filename=kubernetes/config/east
# Delete: kubectl delete --filename=kubernetes/config/east
echo "=================================================="
sleep 30
