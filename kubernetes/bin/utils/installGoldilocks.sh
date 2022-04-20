#!/usr/bin/env bash
set -e

echo "# Install Kubernetes Metrics Server"
# https://github.com/kubernetes-sigs/metrics-server#installation
# The k3d has metrics-server.kube-system, however, it failed verifying for next step.
kubectl apply --filename=https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
# HA: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/high-availability.yaml
echo "=================================================="

echo "# Verify Kubernetes Metrics Server"
kubectl get apiservice v1beta1.metrics.k8s.io
echo "=================================================="

echo "# Install Kubernetes Vertical Pod Autoscaler"
# https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler#install-command
./submodules/autoscaler/vertical-pod-autoscaler/hack/vpa-up.sh
# Delete: ./submodules/autoscaler/vertical-pod-autoscaler/hack/vpa-down.sh
echo "=================================================="

echo "# Install Goldilocks"
# https://goldilocks.docs.fairwinds.com/installation/#installation-2
helm repo add fairwinds-stable https://charts.fairwinds.com/stable
kubectl create namespace goldilocks
helm install goldilocks --namespace=goldilocks fairwinds-stable/goldilocks --set=dashboard.service.port=34617
# helm uninstall goldilocks --namespace=goldilocks
echo "=================================================="

echo "# Label namespaces for Goldilocks"
kubectl label namespace goldilocks goldilocks.fairwinds.com/enabled=true
kubectl label namespace hm goldilocks.fairwinds.com/enabled=true
echo "=================================================="
