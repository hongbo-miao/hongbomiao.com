#!/usr/bin/env bash
set -e

echo "# Install Prometheus and Grafana"
# https://prometheus-operator.dev/docs/prologue/quick-start
kubectl create --filename=kubernetes/manifests/prometheus/setup
kubectl create --filename=kubernetes/manifests/prometheus

# Grafana
# Username: admin
# Password: admin

# Delete:
# kubectl delete \
#   --filename=kubernetes/manifests/prometheus \
#   --filename=kubernetes/manifests/prometheus/setup
echo "=================================================="
