#!/usr/bin/env bash
set -e

echo "# Install ORY Hydra"
kubectl apply --filename=kubernetes/config/ory-hydra/hm-ory-hydra-namespace.yaml

helm repo add ory https://k8s.ory.sh/helm/charts
helm repo update

ORY_HYDRA_SECRET="RGRmdGE5QU1STnZSSnE0Z0FCV"

helm install ory-hydra \
  --namespace=hm-ory-hydra \
  --set="hydra.config.secrets.system={${ORY_HYDRA_SECRET}}" \
  --set="hydra.config.dsn=postgres://admin:passw0rd@postgres-service.hm-postgres:40072/ory_hydra_db" \
  --set="hydra.config.urls.self.issuer=https://my-hydra/" \
  --set="hydra.config.urls.login=https://my-idp/login" \
  --set="hydra.config.urls.consent=https://my-idp/consent" \
  --set="hydra.autoMigrate=true" \
  --set="hydra.dangerousForceHttp=true" \
  --set="hydra.dangerousAllowInsecureRedirectUrls={http://localhost:4445/}" \
  ory/hydra

# Delete:
# helm uninstall ory-hydra --namespace=hm-ory-hydra
echo "=================================================="
