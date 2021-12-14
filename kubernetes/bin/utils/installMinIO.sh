#!/usr/bin/env bash
set -e

echo "# Initialize MinIO operator"
MINIO_PATH="kubernetes/data/minio"
rm -rf "${MINIO_PATH}/log"
rm -rf "${MINIO_PATH}/prometheus"
mkdir "${MINIO_PATH}/log"
mkdir "${MINIO_PATH}/prometheus"
kubectl minio init --cluster-domain="west.k8s-hongbomiao.com"
echo "=================================================="

echo "# Check MinIO operator"
kubectl rollout status deployment/minio-operator --namespace=minio-operator
kubectl rollout status deployment/console --namespace=minio-operator
echo "=================================================="

echo "# Deploy MinIO tenant"
kubectl minio proxy --namespace=minio-operator &
# http://localhost:9090

kubectl apply --filename=kubernetes/config/minio
kubectl apply --kustomize=kubernetes/manifests/minio/tenant-tiny
# kubectl port-forward service/storage-tiny-console --namespace=tenant-tiny 9443:9443
# https://localhost:9443
# Username: minio123
# Password: minio
echo "=================================================="
sleep 30
