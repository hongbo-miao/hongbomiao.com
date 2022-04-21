#!/usr/bin/env bash
set -e

echo "# Install hm-streaming"
flink run-application \
  --target kubernetes-application \
  -Dkubernetes.namespace=hm-flink \
  -Dkubernetes.cluster-id=hm-flink-cluster \
  -Dkubernetes.container.image=hongbomiao/hm-streaming:latest \
  -Dkubernetes.container.image.pull-policy=Always \
  -Dkubernetes.jobmanager.service-account=flink-serviceaccount \
  local:///opt/flink/usrlib/streaming-0.1.jar

# Delete:
# kubectl delete deployment/hm-flink-cluster --namespace=hm-flink
echo "=================================================="
