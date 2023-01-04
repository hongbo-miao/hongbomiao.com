#!/usr/bin/env bash
set -e

echo "# Install Flink"
kubectl apply --filename=kubernetes/manifests/flink/hm-flink-namespace.yaml
kubectl apply --filename=kubernetes/manifests/flink
echo "=================================================="

echo "# Install Redis"
source kubernetes/bin/utils/install_redis.sh
echo "=================================================="

echo "# Install hm-streaming"
kubectl apply --filename=kubernetes/manifests/flink/hm-flink-namespace.yaml
flink run-application \
  --target kubernetes-application \
  -Dkubernetes.namespace=hm-flink \
  -Dkubernetes.cluster-id=hm-flink-cluster \
  -Dkubernetes.container.image=ghcr.io/hongbo-miao/hm-streaming:latest \
  -Dkubernetes.container.image.pull-policy=Always \
  -Dkubernetes.jobmanager.service-account=flink-serviceaccount \
  local:///opt/flink/usrlib/streaming-0.1.jar

# Delete:
# kubectl delete deployment/hm-flink-cluster --namespace=hm-flink
echo "=================================================="
