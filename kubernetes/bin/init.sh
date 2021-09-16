#!/usr/bin/env bash

set -e

# kubectl-minio
wget https://github.com/minio/operator/releases/download/v4.2.7/kubectl-minio_4.2.7_darwin_amd64 -O kubectl-minio
chmod +x kubectl-minio
mv kubectl-minio /usr/local/bin/
kubectl minio version
