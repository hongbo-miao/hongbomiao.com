#!/usr/bin/env bash
set -e

echo "# Uninstall Rasa Open Source"
helm uninstall hm-release --namespace=hm-rasa
kubectl delete --filename=kubernetes/manifests/rasa/hm-rasa-namespace.yaml
echo "=================================================="

echo "# Uninstall Rasa Action Server"
helm uninstall hm-release --namespace=hm-rasa-action-server
kubectl delete --filename=kubernetes/manifests/rasa-action-server/hm-rasa-action-server-namespace.yaml
echo "=================================================="
