#!/usr/bin/env bash
set -e

# echo "# Start minikube"
# minikube start --driver=hyperkit --cpus=2 --memory=8g
# echo "=================================================="

echo "# Remove OPA"
perl -p0i -e 's/# ---- OPA BEGIN ----.*?# ---- OPA END ----//sg' manifests/west/graphql-server-deployment.yaml
echo "=================================================="

echo "# Remove Elastic APM"
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/manifests/west/graphql-server-deployment.yaml
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/manifests/west/grpc-server-deployment.yaml
rm -f kubernetes/manifests/west/elastic-apm-configmap.yaml
rm -f kubernetes/manifests/west/elastic-apm-pv.yaml
rm -f kubernetes/manifests/west/elastic-apm-pvc.yaml
echo "=================================================="

echo "# Install the app"
kubectl apply --filename=kubernetes/manifests/west/hm-namespace.yaml
kubectl apply --filename=kubernetes/manifests/west --selector=app.kubernetes.io/name=graphql-server
kubectl apply --filename=kubernetes/manifests/west --selector=app.kubernetes.io/name=grpc-server
echo "=================================================="
