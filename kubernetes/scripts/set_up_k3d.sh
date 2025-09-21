#!/usr/bin/env bash
set -e

echo "# Setup K3d"
k3d cluster create west --config=k3d/west-cluster-config.yaml
k3d cluster create east --config=k3d/east-cluster-config.yaml
k3d cluster create dev --config=k3d/dev-cluster-config.yaml
# k3d cluster delete west
# k3d cluster delete east
# k3d cluster delete dev

kubectl config use-context k3d-west
# kubectl config use-context k3d-east
# kubectl config use-context k3d-dev
echo "=================================================="

echo "# Install Ingress"
helm upgrade \
  ingress-nginx \
  ingress-nginx \
  --install \
  --repo=https://kubernetes.github.io/ingress-nginx \
  --namespace=ingress-nginx \
  --create-namespace
echo "=================================================="
