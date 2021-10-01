#!/usr/bin/env bash

set -e


# Install the app by Argo CD
echo "# Install the app"
kubectl apply --filename=kubernetes/config/east/hm-namespace.yaml
kubectl apply --filename=kubernetes/config/east
# Delete: kubectl delete --filename=kubernetes/config/east
echo "=================================================="
sleep 30
