#!/usr/bin/env bash
set -e

echo "# Install KubeRay operator"
# https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html
helm upgrade \
  kuberay-operator \
  kuberay-operator \
  --install \
  --repo=https://ray-project.github.io/kuberay-helm \
  --namespace=hm-ray \
  --create-namespace
# helm uninstall kuberay-operator --namespace=hm-ray

echo "# Install Ray cluster"
helm upgrade \
  ray-cluster \
  ray-cluster \
  --install \
  --repo=https://ray-project.github.io/kuberay-helm \
  --namespace=hm-ray \
  --create-namespace \
  --values=kubernetes/manifests/ray/helm/my-values.yaml
# helm uninstall ray-cluster --namespace=hm-ray
# kubectl delete namespace hm-ray
echo "=================================================="
