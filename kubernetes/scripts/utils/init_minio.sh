#!/usr/bin/env bash
set -e

echo "# kubectl-minio"
# Docs: https://github.com/minio/operator
# Version: https://github.com/minio/operator/releases
wget --no-verbose --output-document=kubectl-minio https://github.com/minio/operator/releases/download/v4.2.7/kubectl-minio_4.2.7_darwin_amd64
chmod +x kubectl-minio
mv kubectl-minio /usr/local/bin/
kubectl minio version
echo "=================================================="

echo "# Install mc"
# Code: https://github.com/minio/mc
# Docs: https://docs.min.io/docs/minio-client-quickstart-guide.html
brew install minio/stable/mc
echo "=================================================="
