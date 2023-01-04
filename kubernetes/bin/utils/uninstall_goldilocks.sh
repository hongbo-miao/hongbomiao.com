#!/usr/bin/env bash
set -e

echo "# Unlabel namespaces for Goldilocks"
kubectl label namespace goldilocks goldilocks.fairwinds.com/enabled-
kubectl label namespace hm goldilocks.fairwinds.com/enabled-
echo "=================================================="

echo "# Uninstall Goldilocks"
# https://goldilocks.docs.fairwinds.com/installation/#installation-2
helm uninstall goldilocks --namespace=hm-goldilocks
kubectl delete namespace hm-goldilocks
echo "=================================================="

echo "# Uninstall Kubernetes Vertical Pod Autoscaler"
./submodules/autoscaler/vertical-pod-autoscaler/hack/vpa-down.sh
echo "=================================================="
