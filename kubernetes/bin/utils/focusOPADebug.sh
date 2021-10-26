#!/usr/bin/env bash

set -e

# Turn on debug mode
perl -p0i -e 's/is_debug=false/is_debug=true/s' kubernetes/bin/setup.sh

# Update opal-server replicas from 3 to 1
perl -p0i -e 's/replicas: 3/replicas: 1/s' kubernetes/config/west/opal-server-deployment.yaml

# Remove Elastic APM
# perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/config-server-deployment.yaml
# perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/decision-logger-deployment.yaml
# perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/graphql-server-deployment.yaml
# rm -f kubernetes/config/west/elastic-apm-configmap.yaml
# rm -f kubernetes/config/west/elastic-apm-pv.yaml
# rm -f kubernetes/config/west/elastic-apm-pvc.yaml

# Remove services
rm -f kubernetes/config/west/flink-clusterrolebinding.yaml
rm -f kubernetes/config/west/flink-serviceaccount.yaml
rm -f kubernetes/config/west/grpc-server-configmap.yaml
rm -f kubernetes/config/west/grpc-server-deployment.yaml
rm -f kubernetes/config/west/grpc-server-service.yaml
rm -f kubernetes/config/west/grpc-server-trafficsplit.yaml
rm -f kubernetes/config/west/hm-cnn-namespace.yaml
rm -f kubernetes/config/west/hm-flink-namespace.yaml
rm -f kubernetes/config/west/hm-redis-namespace.yaml
rm -f kubernetes/config/west/model-server-pv.yaml
rm -f kubernetes/config/west/model-server-pvc.yaml
rm -f kubernetes/config/west/torchserve-deployment.yaml
rm -f kubernetes/config/west/torchserve-service.yaml
rm -f kubernetes/config/west/redis-leader-service.yaml
rm -f kubernetes/config/west/redis-leader-deployment.yaml
