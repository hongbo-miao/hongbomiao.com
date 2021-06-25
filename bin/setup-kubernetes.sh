#!/usr/bin/env bash

set -e

# 1. Start minikube
minikube config set cpus 4
minikube config set memory 8192
minikube start


# 2. Install Linkerd
linkerd check --pre

linkerd install | kubectl apply -f -
linkerd check

linkerd viz install | kubectl apply -f -
linkerd viz check

linkerd jaeger install | kubectl apply -f -
linkerd jaeger check


# 3. Install my YAML files
kubectl apply -f kubernetes/*-namespace.yaml
kubectl apply -f kubernetes
