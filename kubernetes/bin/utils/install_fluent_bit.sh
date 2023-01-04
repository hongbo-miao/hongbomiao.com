#!/usr/bin/env bash
set -e

echo "# Install Fluent Bit"
# https://docs.fluentbit.io/manual/installation/kubernetes
kubectl create namespace logging
kubectl apply --filename=https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/fluent-bit-service-account.yaml
kubectl apply --filename=https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/fluent-bit-role-1.22.yaml
kubectl apply --filename=https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/fluent-bit-role-binding-1.22.yaml
kubectl apply --filename=https://raw.githubusercontent.com/fluent/fluent-bit-kubernetes-logging/master/output/elasticsearch/fluent-bit-configmap.yaml
kubectl apply --filename=kubernetes/manifests/fluent-bit
echo "=================================================="
