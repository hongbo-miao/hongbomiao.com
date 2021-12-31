#!/usr/bin/env bash
set -e

echo "# Install Rasa"
kubectl create namespace hm-rasa
helm repo add rasa https://helm.rasa.com

helm install \
  --namespace=hm-rasa \
  --values=kubernetes/config/rasa/rasa-values.yaml \
  hm-release \
  rasa/rasa
# Delete: helm uninstall --namespace=hm-rasa hm-release
