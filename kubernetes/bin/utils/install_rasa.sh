#!/usr/bin/env bash
set -e

echo "# Install Rasa Action Server"
helm upgrade \
  hm-release \
  rasa-action-server \
  --install \
  --repo=https://helm.rasa.com \
  --namespace=hm-rasa-action-server \
  --create-namespace \
  --values=kubernetes/manifests/rasa-action-server/helm/my-values.yaml
kubectl apply --filename=kubernetes/manifests/rasa-action-server/hm-rasa-action-server-namespace.yaml
# helm uninstall hm-release --namespace=hm-rasa-action-server
# kubectl delete namespace hm-rasa-action-server
echo "=================================================="

echo "# Install Rasa Open Source"
helm upgrade \
  hm-release \
  rasa \
  --install \
  --repo=https://helm.rasa.com \
  --namespace=hm-rasa \
  --create-namespace \
  --values=kubernetes/manifests/rasa/helm/my-values.yaml
# helm uninstall hm-release --namespace=hm-rasa
# kubectl delete namespace hm-rasa
echo "=================================================="
