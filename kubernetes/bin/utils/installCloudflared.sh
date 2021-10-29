#!/usr/bin/env bash

set -e


echo "# Install cloudflared"
kubectl apply --filename=kubernetes/config/cloudflared/hm-clouflared-namespace.yaml
kubectl apply --filename=kubernetes/config/cloudflared
# kubectl delete --filename=kubernetes/config/cloudflared

kubectl create secret generic tunnel-credentials \
  --from-file=credentials.json=/Users/homiao/.cloudflared/7aca8dbc-634c-43b8-9a5d-51d84370ed02.json \
  --namespace=hm-cloudflared
echo "=================================================="
