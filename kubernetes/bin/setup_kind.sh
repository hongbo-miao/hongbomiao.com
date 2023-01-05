#!/usr/bin/env bash
set -e

echo "# Setup kind"
kind create cluster --name=west --config=kind/west-cluster-config.yaml
kind create cluster --name=east --config=kubernetes/kind/east-cluster-config.yaml
# Delete: kind delete cluster --name=west
# Delete: kind delete cluster --name=east

kubectl config use-context kind-west
# kubectl config use-context kind-east

INGRESS_VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
kubectl apply --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_VERSION}/deploy/static/provider/kind/deploy.yaml"
# Local: kubectl apply --filename=kubernetes/manifests-raw/ingress-nginx.yaml
# Delete: kubectl delete --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_VERSION}/deploy/static/provider/kind/deploy.yaml"
echo "=================================================="
