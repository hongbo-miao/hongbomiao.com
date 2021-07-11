#!/usr/bin/env bash

set -e

kind create cluster --name=west --config=kubernetes/kind/west-cluster-config.yaml
kind create cluster --name=east --config=kubernetes/kind/east-cluster-config.yaml
# kind delete cluster --name=west
# kind delete cluster --name=east

kubectl config use-context kind-west
# kubectl config use-context kind-east

VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
kubectl apply --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/kind/deploy.yaml"
# kubectl delete --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/kind/deploy.yaml"
