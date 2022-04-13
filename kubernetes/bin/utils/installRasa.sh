#!/usr/bin/env bash
set -e

echo "# Install Rasa Action Server"
kubectl apply --filename=kubernetes/config/rasa/hm-rasa-action-server-namespace.yaml
helm repo add rasa https://helm.rasa.com

helm install hm-release \
  --namespace=hm-rasa-action-server \
  --values=kubernetes/config/rasa/rasa-action-server/values.yaml \
  rasa/rasa-action-server

# Upgrade:
# helm upgrade hm-release \
#   --namespace=hm-rasa-action-server \
#   --values=kubernetes/config/rasa/rasa-action-server/values.yaml \
#   rasa/rasa-action-server

# Delete:
# helm uninstall --namespace=hm-rasa-action-server hm-release
echo "=================================================="

echo "# Install Rasa Open Source"
kubectl apply --filename=kubernetes/config/rasa/hm-rasa-namespace.yaml

helm install hm-release \
  --namespace=hm-rasa \
  --values=kubernetes/config/rasa/rasa/values.yaml \
  rasa/rasa

# Upgrade:
# helm upgrade hm-release \
#   --namespace=hm-rasa \
#   --reuse-values \
#   --values=kubernetes/config/rasa/rasa/values.yaml \
#   rasa/rasa

# Delete:
# helm uninstall --namespace=hm-rasa hm-release
echo "=================================================="
