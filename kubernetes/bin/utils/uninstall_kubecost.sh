#!/usr/bin/env bash
set -e

echo "# Uninstall Kubecost"
helm uninstall kubecost --namespace=hm-kubecost
kubectl delete namespace hm-kubecost
echo "=================================================="
