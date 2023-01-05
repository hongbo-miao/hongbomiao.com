#!/usr/bin/env bash
set -e

echo "# Install hm-alpine"
kubectl apply --filename=kubernetes/manifests/hm-alpine/hm-alpine-namespace.yaml
kubectl apply --filename=kubernetes/manifests/hm-alpine

# Delete:
# kubectl delete --filename=kubernetes/manifests/hm-alpine
echo "=================================================="
