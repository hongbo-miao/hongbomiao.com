#!/usr/bin/env bash
set -e

echo "# Turn on debug mode"
perl -p0i -e 's/is_debug=false/is_debug=true/s' kubernetes/bin/setup.sh
echo "=================================================="

echo "# Update opal-server replicas from 3 to 1"
perl -p0i -e 's/replicas: 3/replicas: 1/s' kubernetes/manifests/west/opal-server-deployment.yaml
echo "=================================================="

# echo "# Remove Elastic APM"
# perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/manifests/west/config-loader-deployment.yaml
# perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/manifests/west/decision-logger-deployment.yaml
# perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/manifests/west/graphql-server-deployment.yaml
# rm -f kubernetes/manifests/west/elastic-apm-configmap.yaml
# rm -f kubernetes/manifests/west/elastic-apm-pv.yaml
# rm -f kubernetes/manifests/west/elastic-apm-pvc.yaml
# echo "=================================================="

echo "# Remove services"
rm -f kubernetes/manifests/west/grpc-server-configmap.yaml
rm -f kubernetes/manifests/west/grpc-server-deployment.yaml
rm -f kubernetes/manifests/west/grpc-server-service.yaml
rm -f kubernetes/manifests/west/grpc-server-trafficsplit.yaml
rm -f kubernetes/manifests/west/hm-cnn-namespace.yaml
rm -f kubernetes/manifests/west/model-server-pv.yaml
rm -f kubernetes/manifests/west/model-server-pvc.yaml
rm -f kubernetes/manifests/west/torchserve-deployment.yaml
rm -f kubernetes/manifests/west/torchserve-service.yaml
echo "=================================================="
