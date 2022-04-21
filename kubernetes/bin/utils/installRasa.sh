#!/usr/bin/env bash
set -e

echo "# Install Rasa Action Server"
kubectl apply --filename=kubernetes/manifests/rasa/hm-rasa-action-server-namespace.yaml
helm repo add rasa https://helm.rasa.com

helm install \
  hm-release \
  rasa/rasa-action-server \
  --namespace=hm-rasa-action-server \
  --values=kubernetes/manifests/rasa/rasa-action-server/values.yaml

# Upgrade:
# helm upgrade \
#   hm-release \
#   rasa/rasa-action-server \
#   --namespace=hm-rasa-action-server \
#   --values=kubernetes/manifests/rasa/rasa-action-server/values.yaml

# Delete:
# helm uninstall hm-release --namespace=hm-rasa-action-server
echo "=================================================="

echo "# Install Rasa Open Source"
kubectl apply --filename=kubernetes/manifests/rasa/hm-rasa-namespace.yaml

helm install \
  hm-release \
  rasa/rasa \
  --namespace=hm-rasa \
  --values=kubernetes/manifests/rasa/rasa/values.yaml

# Upgrade:
# helm upgrade \
#   hm-release \
#   rasa/rasa \
#   --namespace=hm-rasa \
#   --reuse-values \
#   --values=kubernetes/manifests/rasa/rasa/values.yaml

# Delete:
# helm uninstall hm-release --namespace=hm-rasa
echo "=================================================="
