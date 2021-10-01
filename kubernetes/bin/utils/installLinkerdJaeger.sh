#!/usr/bin/env bash

set -e


# Install Linkerd Jaeger
echo "# Install Linkerd Jaeger"
linkerd jaeger install | \
  kubectl apply --filename=-
echo "=================================================="
sleep 30

# linkerd jaeger dashboard --context=k3d-west


# Check Linkerd Jaeger
echo "# Check Linkerd Jaeger"
linkerd jaeger check
echo "=================================================="
