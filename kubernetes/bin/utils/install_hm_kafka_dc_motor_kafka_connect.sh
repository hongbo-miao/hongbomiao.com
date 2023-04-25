#!/usr/bin/env bash
set -e

echo "# Install hm-kafka-dc-motor-kafka-connect"
kubectl apply --filename=kubernetes/manifests/hm-kafka/dc-motor-kafka-connect
# kubectl delete --filename=kubernetes/manifests/hm-kafka/dc-motor-kafka-connect
echo "=================================================="
