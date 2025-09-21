#!/usr/bin/env bash
set -e

echo "# Install Linkerd Jaeger"
linkerd jaeger install | \
  kubectl apply --filename=-
sleep 30
echo "=================================================="

# linkerd jaeger dashboard --context=k3d-west

echo "# Check Linkerd Jaeger"
linkerd jaeger check
echo "=================================================="
