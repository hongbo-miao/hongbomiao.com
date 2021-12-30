#!/usr/bin/env bash
set -e

echo "# Install cloudflared"
kubectl apply --filename=kubernetes/config/cloudflared/hm-clouflared-namespace.yaml
kubectl apply --filename=kubernetes/config/cloudflared
# kubectl delete --filename=kubernetes/config/cloudflared

kubectl create secret generic tunnel-credentials \
  --from-file=credentials.json=/Users/homiao/.cloudflared/afccc94c-0065-4e40-832a-a00b1940faaf.json \
  --namespace=hm-cloudflared
echo "=================================================="
