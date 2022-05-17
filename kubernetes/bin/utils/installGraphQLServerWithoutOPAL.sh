#!/usr/bin/env bash
set -e

echo "# Remove OPA"
perl -p0i -e 's/# ---- OPA BEGIN ----.*?# ---- OPA END ----//sg' kubernetes/manifests/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Remove Elastic APM"
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/manifests/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Install GraphQL Server without OPAL"
kubectl apply --filename=kubernetes/manifests/west/hm-namespace.yaml
kubectl apply --filename=kubernetes/manifests/west --selector=app.kubernetes.io/name=graphql-server
echo "=================================================="
