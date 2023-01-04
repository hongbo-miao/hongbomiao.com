#!/usr/bin/env bash
set -e

echo "# Uninstall Telegraf"
kubectl delete --filename=kubernetes/manifests/telegraf
kubectl delete secret hm-telegraf-secret --namespace=hm-telegraf
echo "=================================================="
