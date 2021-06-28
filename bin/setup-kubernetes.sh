#!/usr/bin/env bash

set -e

# Start minikube
minikube config set cpus 4
minikube config set memory 8192
minikube start


# Install Linkerd
linkerd check --pre

linkerd install | kubectl apply --filename=-
linkerd check

linkerd viz install | kubectl apply --filename=-
linkerd viz check

linkerd jaeger install | kubectl apply --filename=-
linkerd jaeger check


# Install Argo CD
kubectl create namespace argocd
kubectl apply --namespace=argocd --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

for deploy in "dex-server" "redis" "repo-server" "server"; \
  do kubectl --namespace=argocd rollout status deploy/argocd-${deploy}; \
done


# Install the app by Argo CD
kubectl apply --filename=argocd/hm-application.yaml
kubectl port-forward svc/argocd-server -n argocd 8080:443
argocd login localhost:8080
argocd app sync hm-application


# Install the app by Kubernetes files
# kubectl apply --filename=kubernetes/*-namespace.yaml
# kubectl apply --filename=kubernetes
