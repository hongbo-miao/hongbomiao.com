#!/usr/bin/env bash
set -e

echo "# Install HM API Python"
kubectl apply --filename=kubernetes/manifests/hm-api-python/hm-api-python-namespace.yaml
kubectl apply --filename=kubernetes/manifests/hm-api-python

# Delete:
# kubectl delete --filename=kubernetes/manifests/hm-api-python
echo "=================================================="
