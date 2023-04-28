#!/usr/bin/env bash
set -e

echo "# Install pgAdmin"
# https://artifacthub.io/packages/helm/runix/pgadmin4
helm upgrade \
  pgadmin \
  pgadmin4 \
  --install \
  --repo=https://helm.runix.net \
  --namespace=hm-pgadmin \
  --create-namespace \
  --values=kubernetes/manifests/pgadmin/helm/my-values.yaml
# helm uninstall pgadmin --namespace=hm-pgadmin
# kubectl delete namespace hm-pgadmin
echo "=================================================="
