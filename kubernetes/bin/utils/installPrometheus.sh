#!/usr/bin/env bash
set -e

echo "# Install Prometheus and Grafana"
# https://prometheus-operator.dev/docs/prologue/quick-start
kubectl create --filename=kubernetes/config/prometheus/setup
kubectl create --filename=kubernetes/config/prometheus

# Grafana
# Username: admin
# Password: admin

# Delete:
# kubectl delete \
#   --filename=kubernetes/config/prometheus \
#   --filename=kubernetes/config/prometheus/setup
echo "=================================================="
