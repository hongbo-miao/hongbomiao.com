#!/usr/bin/env bash

set -e

# Start minikube
echo "Start minikube"
minikube config set cpus 4
minikube config set memory 8192
minikube start --driver=hyperkit
minikube addons enable ingress
minikube mount "$HOME/Clouds/Git/hongbomiao.com/kubernetes/data:/data"


# Install Linkerd
linkerd check --pre

echo "Install Linkerd"
linkerd install --disable-heartbeat | kubectl apply --filename=-
# In production, use
# linkerd install --disable-heartbeat --ha | kubectl apply --filename=-
linkerd check

echo "Install Linkerd Viz"
linkerd viz install | kubectl apply --filename=-
linkerd viz check

echo "Install Linkerd Jaeger"
linkerd jaeger install | kubectl apply --filename=-
linkerd jaeger check


# Patch Ingress
kubectl patch configmap ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes-patch/ingress-nginx-controller-configmap-patch.yaml)"
kubectl patch deployment ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes-patch/ingress-nginx-controller-deployment-patch.yaml)"


# Install Argo CD
echo "Install Argo CD"
kubectl create namespace argocd
kubectl apply --namespace=argocd --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

for deploy in "dex-server" "redis" "repo-server" "server"; \
  do kubectl --namespace=argocd rollout status deploy/argocd-${deploy}; \
done


# Install the app by Argo CD
echo "Install the app"
kubectl apply --filename=argocd/hm-application.yaml
kubectl port-forward svc/argocd-server -n argocd 8080:443
argocd login localhost:8080
argocd app sync hm-application --local=kubernetes/config


# Install the app by Kubernetes files
# echo "Install the app"
# kubectl apply --filename=kubernetes/config/hm-namespace.yaml
# kubectl apply --filename=kubernetes/config
