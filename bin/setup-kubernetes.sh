#!/usr/bin/env bash

set -e

# Start minikube
minikube config set cpus 4
minikube config set memory 8192
minikube start

# Install Linkerd
linkerd install | kubectl apply -f -
linkerd viz install | kubectl apply -f -
linkerd jaeger install | kubectl apply -f -

# Deploy
kubectl apply -f helm-chart/hm-chart/templates/*namespace.yaml
kubectl apply -f helm-chart/hm-chart/templates
kubectl get deployments --namespace=hm --output=yaml | linkerd inject - | kubectl apply -f -
