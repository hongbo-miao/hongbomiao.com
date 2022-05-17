#!/usr/bin/env bash
set -e

echo "# Uninstall Temporal"
helm uninstall temporal --namespace=hm-temporal
kubectl delete job temporal-schema-setup --namespace=hm-temporal
kubectl delete namespace hm-temporal
echo "=================================================="
