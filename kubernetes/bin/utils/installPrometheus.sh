#!/usr/bin/env bash
set -e

echo "# Install Prometheus"
# https://prometheus-operator.dev/docs/prologue/quick-start/
kubectl create --filename=kubernetes/config/prometheus/setup
kubectl create --filename=kubernetes/config/prometheus

# Delete:
# kubectl delete \
#   --filename=kubernetes/config/prometheus \
#   --filename=kubernetes/config/prometheus/setup
echo "=================================================="
