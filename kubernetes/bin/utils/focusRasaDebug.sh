#!/usr/bin/env bash
set -e

# echo "# Setup K3d"
# k3d cluster create dev --config=kubernetes/k3d/dev-cluster-config.yaml
# # Delete: k3d cluster delete dev
# echo "=================================================="

echo "# Remove OPA"
perl -p0i -e 's/# ---- OPA BEGIN ----.*?# ---- OPA END ----//sg' kubernetes/manifests/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Remove Elastic APM"
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/manifests/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Remove services"
rm -f kubernetes/manifests/west/config-server-configmap.yaml
rm -f kubernetes/manifests/west/config-server-deployment.yaml
rm -f kubernetes/manifests/west/config-server-pv.yaml
rm -f kubernetes/manifests/west/config-server-pvc.yaml
rm -f kubernetes/manifests/west/config-server-service.yaml
rm -f kubernetes/manifests/west/decision-logger-configmap.yaml
rm -f kubernetes/manifests/west/decision-logger-deployment.yaml
rm -f kubernetes/manifests/west/decision-logger-service.yaml
rm -f kubernetes/manifests/west/dgraph-alpha-public-service.yaml
rm -f kubernetes/manifests/west/dgraph-alpha-service.yaml
rm -f kubernetes/manifests/west/dgraph-alpha-statefulset.yaml
rm -f kubernetes/manifests/west/dgraph-zero-public-service.yaml
rm -f kubernetes/manifests/west/dgraph-zero-service.yaml
rm -f kubernetes/manifests/west/dgraph-zero-statefulset.yaml
rm -f kubernetes/manifests/west/elastic-apm-configmap.yaml
rm -f kubernetes/manifests/west/elastic-apm-pv.yaml
rm -f kubernetes/manifests/west/elastic-apm-pvc.yaml
rm -f kubernetes/manifests/west/grpc-server-configmap.yaml
rm -f kubernetes/manifests/west/grpc-server-deployment.yaml
rm -f kubernetes/manifests/west/grpc-server-service.yaml
rm -f kubernetes/manifests/west/grpc-server-trafficsplit.yaml
rm -f kubernetes/manifests/west/hm-cnn-namespace.yaml
rm -f kubernetes/manifests/west/hm-dgraph-namespace.yaml
rm -f kubernetes/manifests/west/hm-opa-namespace.yaml
rm -f kubernetes/manifests/west/hm-postgres-namespace.yaml
rm -f kubernetes/manifests/west/model-server-pv.yaml
rm -f kubernetes/manifests/west/model-server-pvc.yaml
rm -f kubernetes/manifests/west/opa-pv.yaml
rm -f kubernetes/manifests/west/opa-pvc.yaml
rm -f kubernetes/manifests/west/opal-broadcast-channel-configmap.yaml
rm -f kubernetes/manifests/west/opal-broadcast-channel-deployment.yaml
rm -f kubernetes/manifests/west/opal-broadcast-channel-service.yaml
rm -f kubernetes/manifests/west/opal-server-configmap.yaml
rm -f kubernetes/manifests/west/opal-server-deployment.yaml
rm -f kubernetes/manifests/west/opal-server-service.yaml
rm -f kubernetes/manifests/west/torchserve-deployment.yaml
rm -f kubernetes/manifests/west/torchserve-service.yaml
echo "=================================================="

echo "# Install the app"
kubectl apply --filename=kubernetes/manifests/west/hm-namespace.yaml
kubectl apply --filename=kubernetes/manifests/west
echo "=================================================="

echo "# Install Rasa"
source kubernetes/bin/utils/installRasa.sh
echo "=================================================="

echo "# Install cloudflared"
source kubernetes/bin/utils/installCloudflared.sh
echo "=================================================="
