#!/usr/bin/env bash
set -e

echo "# Install Hasura"
kubectl apply --filename=kubernetes/config/hasura/hm-hasura-namespace.yaml
kubectl apply --filename=kubernetes/config/hasura
echo "=================================================="
