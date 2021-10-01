#!/usr/bin/env bash

set -e


# Create clusters
echo "# Create clusters"
k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
k3d cluster create east --config=kubernetes/k3d/east-cluster-config.yaml

# k3d cluster delete west
# k3d cluster delete east

# kubectl config use-context k3d-west
# kubectl config use-context k3d-east

sleep 30
echo "=================================================="
