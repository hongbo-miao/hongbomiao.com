#!/usr/bin/env bash
set -e

echo "# Install Argo CD"
kubectl create namespace argocd
kubectl apply \
  --namespace=argocd \
  --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
# Local:
# kubectl apply \
#   --namespace=argocd \
#   --filename=kubernetes/manifests-raw/argocd/argocd.yaml
# Delete:
# kubectl delete \
#   --namespace=argocd \
#   --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
echo "=================================================="
sleep 30

echo "# Check Argo CD"
for d in dex-server redis repo-server server; do
  kubectl rollout status deployment/argocd-${d} --namespace=argocd
done
echo "=================================================="
