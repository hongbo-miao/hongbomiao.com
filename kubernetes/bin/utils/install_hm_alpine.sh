#!/usr/bin/env bash
set -e

echo "# Install HM Alpine"
kubectl apply --filename=kubernetes/manifests/hm-alpine/hm-alpine-namespace.yaml
kubectl apply --filename=kubernetes/manifests/hm-alpine

# Delete:
# kubectl delete --filename=kubernetes/manifests/hm-alpine
echo "=================================================="
