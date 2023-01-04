#!/usr/bin/env bash
set -e

echo "# Install the app"
kubectl port-forward service/argocd-server --namespace=argocd 31026:443 &
ARGOCD_PASSWORD=$(kubectl get secret argocd-initial-admin-secret \
  --namespace=argocd \
  --output=jsonpath="{.data.password}" | \
  base64 -d && echo)
argocd login localhost:31026 --username=admin --password="${ARGOCD_PASSWORD}" --insecure
kubectl apply --filename=kubernetes/manifests/argocd/hm-application.yaml
argocd app sync hm-application --grpc-web --local=kubernetes/manifests/west
pgrep kubectl | xargs kill -9
# Delete: argocd app delete hm-application --yes
echo "=================================================="
