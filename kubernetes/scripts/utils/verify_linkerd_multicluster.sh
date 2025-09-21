#!/usr/bin/env bash
set -e

# Verify Linkerd multicluster setup correctly by checking the endpoints in west
# and verify that they match the gateway’s public IP address in east
kubectl get endpoint grpc-server-service-k3d-east \
  --context=k3d-west \
  --namespace=hm \
  --output="custom-columns=ENDPOINT_IP:.subsets[*].addresses[*].ip"
kubectl linkerd-multicluster get service linkerd-gateway \
  --context=k3d-east \
  --namespace=hm \
  --output="custom-columns=GATEWAY_IP:.status.loadBalancer.ingress[*].ip"
