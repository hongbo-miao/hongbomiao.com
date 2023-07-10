#!/usr/bin/env bash
set -e

echo "# Install Kubeflow"
# https://www.kubeflow.org/docs/components/pipelines/v2/installation/quickstart/
# https://github.com/kubeflow/pipelines/releases?q=Version
export PIPELINE_VERSION=2.0.0
kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
kubectl wait crd/applications.app.k8s.io --for=condition=established --timeout=60s
kubectl apply --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"
# kubectl delete --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=${PIPELINE_VERSION}"
# kubectl delete --kustomize="github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"
# kubectl delete namespace kubeflow
echo "=================================================="

echo "# Create a PyTorch training job"
kubectl create --filename=https://raw.githubusercontent.com/kubeflow/training-operator/master/examples/pytorch/simple.yaml
# kubectl get pytorchjobs --namespace=kubeflow
# kubectl get pytorchjobs pytorch-simple --output=yaml
echo "=================================================="
