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

INGRESS_VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
kubectl apply --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_VERSION}/deploy/static/provider/cloud/deploy.yaml"
# Local: kubectl apply --filename=kubernetes/manifests-raw/ingress-nginx.yaml
echo "=================================================="
