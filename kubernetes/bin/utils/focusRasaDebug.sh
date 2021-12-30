#!/usr/bin/env bash
set -e

# echo "# Setup K3d"
# k3d cluster create dev --config=kubernetes/k3d/dev-cluster-config.yaml
# # Delete: k3d cluster delete dev
# echo "=================================================="

echo "# Remove OPA"
perl -p0i -e 's/# ---- OPA BEGIN ----.*?# ---- OPA END ----//sg' kubernetes/config/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Remove Elastic APM"
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Remove services"
rm -f kubernetes/config/west/config-server-configmap.yaml
rm -f kubernetes/config/west/config-server-deployment.yaml
rm -f kubernetes/config/west/config-server-pv.yaml
rm -f kubernetes/config/west/config-server-pvc.yaml
rm -f kubernetes/config/west/config-server-service.yaml
rm -f kubernetes/config/west/decision-logger-configmap.yaml
rm -f kubernetes/config/west/decision-logger-deployment.yaml
rm -f kubernetes/config/west/decision-logger-service.yaml
rm -f kubernetes/config/west/dgraph-alpha-public-service.yaml
rm -f kubernetes/config/west/dgraph-alpha-service.yaml
rm -f kubernetes/config/west/dgraph-alpha-statefulset.yaml
rm -f kubernetes/config/west/dgraph-zero-public-service.yaml
rm -f kubernetes/config/west/dgraph-zero-service.yaml
rm -f kubernetes/config/west/dgraph-zero-statefulset.yaml
rm -f kubernetes/config/west/elastic-apm-configmap.yaml
rm -f kubernetes/config/west/elastic-apm-pv.yaml
rm -f kubernetes/config/west/elastic-apm-pvc.yaml
rm -f kubernetes/config/west/flink-clusterrolebinding.yaml
rm -f kubernetes/config/west/flink-serviceaccount.yaml
rm -f kubernetes/config/west/grpc-server-configmap.yaml
rm -f kubernetes/config/west/grpc-server-deployment.yaml
rm -f kubernetes/config/west/grpc-server-service.yaml
rm -f kubernetes/config/west/grpc-server-trafficsplit.yaml
rm -f kubernetes/config/west/hm-cnn-namespace.yaml
rm -f kubernetes/config/west/hm-dgraph-namespace.yaml
rm -f kubernetes/config/west/hm-flink-namespace.yaml
rm -f kubernetes/config/west/hm-opa-namespace.yaml
rm -f kubernetes/config/west/hm-postgres-namespace.yaml
rm -f kubernetes/config/west/hm-redis-namespace.yaml
rm -f kubernetes/config/west/model-server-pv.yaml
rm -f kubernetes/config/west/model-server-pvc.yaml
rm -f kubernetes/config/west/opa-pv.yaml
rm -f kubernetes/config/west/opa-pvc.yaml
rm -f kubernetes/config/west/opal-broadcast-channel-configmap.yaml
rm -f kubernetes/config/west/opal-broadcast-channel-deployment.yaml
rm -f kubernetes/config/west/opal-broadcast-channel-service.yaml
rm -f kubernetes/config/west/opal-server-configmap.yaml
rm -f kubernetes/config/west/opal-server-deployment.yaml
rm -f kubernetes/config/west/opal-server-service.yaml
rm -f kubernetes/config/west/postgres-configmap.yaml
rm -f kubernetes/config/west/postgres-service.yaml
rm -f kubernetes/config/west/postgres-statefulset.yaml
rm -f kubernetes/config/west/redis-leader-deployment.yaml
rm -f kubernetes/config/west/redis-leader-service.yaml
rm -f kubernetes/config/west/torchserve-deployment.yaml
rm -f kubernetes/config/west/torchserve-service.yaml
echo "=================================================="

echo "# Install the app"
kubectl apply --filename=kubernetes/config/west/hm-namespace.yaml
kubectl apply --filename=kubernetes/config/west
echo "=================================================="

echo "# Install Rasa"
source kubernetes/bin/utils/installRasa.sh
echo "=================================================="

echo "# Install cloudflared"
source kubernetes/bin/utils/installCloudflared.sh
echo "=================================================="
