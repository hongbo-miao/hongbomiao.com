#!/usr/bin/env bash
set -e

echo "# Install hm-api-python-flask"
kubectl apply --filename=kubernetes/manifests/hm-api-python-flask/hm-api-python-flask-namespace.yaml
kubectl apply --filename=kubernetes/manifests/hm-api-python-flask

# Delete:
# kubectl delete --filename=kubernetes/manifests/hm-api-python-flask
echo "=================================================="
