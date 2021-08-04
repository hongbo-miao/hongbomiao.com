#!/usr/bin/env bash

set -e

k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
k3d cluster create east --config=kubernetes/k3d/east-cluster-config.yaml
#	Delete: k3d cluster delete west
#	Delete: k3d cluster delete east

kubectl config use-context k3d-west
# kubectl config use-context k3d-east

VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
kubectl apply --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/cloud/deploy.yaml"
# Local: kubectl apply --filename=kubernetes/manifests/ingress-nginx.yaml
