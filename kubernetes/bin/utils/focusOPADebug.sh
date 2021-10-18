#!/usr/bin/env bash

set -e


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
rm -f kubernetes/config/west/hm-flink-namespace.yaml
rm -f kubernetes/config/west/hm-redis-namespace.yaml
rm -f kubernetes/config/west/model-server-pv.yaml
rm -f kubernetes/config/west/model-server-pvc.yaml
rm -f kubernetes/config/west/torchserve-deployment.yaml
rm -f kubernetes/config/west/torchserve-service.yaml
rm -f kubernetes/config/west/redis-leader-service.yaml
rm -f kubernetes/config/west/redis-leader-deployment.yaml

perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/decision-logger-deployment.yaml
perl -p0i -e 's/# ---- ELASTIC APM BEGIN ----.*?# ---- ELASTIC APM END ----//sg' kubernetes/config/west/graphql-server-deployment.yaml
