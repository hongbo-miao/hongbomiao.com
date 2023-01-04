#!/usr/bin/env bash
set -e

echo "# Install Linkerd Buoyant"
curl -sL buoyant.cloud/install | sh
linkerd buoyant install | \
  kubectl apply --filename=-
echo "=================================================="
sleep 30
