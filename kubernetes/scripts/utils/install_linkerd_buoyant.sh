#!/usr/bin/env bash
set -e

echo "# Install Linkerd Buoyant"
curl --silent --fail --show-error --location buoyant.cloud/install | sh
linkerd buoyant install | \
  kubectl apply --filename=-
echo "=================================================="
sleep 30
