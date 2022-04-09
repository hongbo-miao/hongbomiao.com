#!/usr/bin/env bash
set -e

# echo "# Start minikube"
# minikube start --driver=hyperkit --cpus=2 --memory=8g
# echo "=================================================="

echo "# Remove OPA"
perl -p0i -e 's/# ---- OPA BEGIN ----.*?# ---- OPA END ----//sg' kubernetes/config/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Remove Elastic APM"
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/graphql-server-deployment.yaml
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/grpc-server-deployment.yaml
rm -f kubernetes/config/west/elastic-apm-configmap.yaml
rm -f kubernetes/config/west/elastic-apm-pv.yaml
rm -f kubernetes/config/west/elastic-apm-pvc.yaml
echo "=================================================="

echo "# Install the app"
kubectl apply --filename=kubernetes/config/west/hm-namespace.yaml
kubectl apply --filename=kubernetes/config/west --selector=app.kubernetes.io/name=graphql-server
kubectl apply --filename=kubernetes/config/west --selector=app.kubernetes.io/name=grpc-server
echo "=================================================="
