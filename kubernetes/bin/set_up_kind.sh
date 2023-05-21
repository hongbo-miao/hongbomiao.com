#!/usr/bin/env bash
set -e

echo "# Setup kind"
kind create cluster --name=west --config=kind/west-cluster-config.yaml
kind create cluster --name=east --config=kubernetes/kind/east-cluster-config.yaml
# Delete: kind delete cluster --name=west
# Delete: kind delete cluster --name=east

kubectl config use-context kind-west
# kubectl config use-context kind-east
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
