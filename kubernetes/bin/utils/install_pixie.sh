#!/usr/bin/env bash
set -e

# https://docs.px.dev/installing-pixie/install-guides/self-hosted-pixie/

echo "# Deploy Pixie Cloud"
git clone https://github.com/pixie-io/pixie.git
cd pixie

# brew install mkcert
mkcert -install

kubectl create namespace plc
./scripts/create_cloud_secrets.sh
kustomize build k8s/cloud_deps/base/elastic/operator | kubectl apply --filename=-
kustomize build k8s/cloud_deps/public | kubectl apply --filename=-
kustomize build k8s/cloud/public/ | kubectl apply --filename=-

# Set up DNS
kubectl get service cloud-proxy-service -n plc
kubectl get service vzconn-service -n plc
go build src/utils/dev_dns_updater/dev_dns_updater.go
./dev_dns_updater --domain-name="dev.withpixie.dev"  --kubeconfig="${HOME}/.kube/config" --n=plc
echo "=================================================="

echo "# Install the Pixie CLI"
export PL_CLOUD_ADDR=dev.withpixie.dev
bash -c "$(curl -fsSL https://withpixie.ai/install.sh)"
echo "=================================================="

echo "# Deploy Pixie"
px deploy --dev_cloud_namespace plc
echo "=================================================="
