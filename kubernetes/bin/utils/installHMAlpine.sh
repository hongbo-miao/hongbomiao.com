#!/usr/bin/env bash
set -e

echo "# Install HM Alpine"
kubectl apply --filename=kubernetes/config/hm-alpine/hm-alpine-namespace.yaml
kubectl apply --filename=kubernetes/config/hm-alpine
# Delete: kubectl delete --filename=kubernetes/config/hm-alpine
echo "=================================================="
