#!/usr/bin/env bash
set -e

echo "# Install cloudflared"
kubectl create secret generic tunnel-credentials \
  --from-file=credentials.json=/Users/hongbo-miao/.cloudflared/c9aa4140-fee8-4862-a479-3c1faacbd816.json \
  --namespace=hm-cloudflared
# kubectl delete secret tunnel-credentials --namespace=hm-cloudflared

kubectl apply --filename=kubernetes/manifests/cloudflared/hm-clouflared-namespace.yaml
kubectl apply --filename=kubernetes/manifests/cloudflared
# kubectl delete --filename=kubernetes/manifests/cloudflared
echo "=================================================="
