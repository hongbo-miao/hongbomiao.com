#!/usr/bin/env bash
set -e

echo "# Install Rasa"
kubectl create namespace rasa
helm repo add rasa https://helm.rasa.com
helm install \
    --namespace rasa \
    --values kubernetes/config/rasa/rasa-values.yaml \
    hm-release \
    rasa/rasa
