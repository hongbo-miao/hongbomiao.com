#!/usr/bin/env bash
set -e

CLUSTERS=("$@")

echo "# Create clusters"
for cluster in "${CLUSTERS[@]}"; do
  k3d cluster create "${cluster}" --config="kubernetes/k3d/${cluster}-cluster-config.yaml"

  # k3d cluster delete west
  # k3d cluster delete east

  # kubectl config use-context k3d-west
  # kubectl config use-context k3d-east
done
sleep 30
echo "=================================================="
