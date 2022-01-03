#!/usr/bin/env bash
set -e

echo "# Uninstall Rasa Open Source"
helm uninstall --namespace=hm-rasa hm-release
kubectl delete --filename=kubernetes/config/rasa/hm-rasa-namespace.yaml
echo "=================================================="

echo "# Uninstall Rasa Action Server"
helm uninstall --namespace=hm-rasa-action-server hm-release
kubectl delete --filename=kubernetes/config/rasa/hm-rasa-action-server-namespace.yaml
echo "=================================================="
