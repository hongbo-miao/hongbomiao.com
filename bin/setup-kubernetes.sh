#!/usr/bin/env bash

set -e

cd helm-chart


# 1. Start minikube
minikube config set cpus 4
minikube config set memory 8192
minikube start


# 2. Install Linkerd
linkerd check --pre

# Add the repo for Linkerd2 stable and edge releases
helm repo add linkerd https://helm.linkerd.io/stable
helm repo add linkerd-edge https://helm.linkerd.io/edge

# https://linkerd.io/2.10/tasks/generate-certificates/#generating-the-certificates-with-step
# Generate trust anchor certificate
CA_CRT=ca.crt
CA_KEY=ca.key
if [ -f "$CA_CRT" ] && [ -f "$CA_KEY" ]; then
  echo "Both ca.crt and ca.key exist."
else
  step certificate create root.linkerd.cluster.local ca.crt ca.key \
    --profile=root-ca --no-password --insecure
fi

# Generate issuer certificate and key
ISSUER_CRT=issuer.crt
ISSUER_KEY=issuer.key
if [ -f "$ISSUER_CRT" ] && [ -f "$ISSUER_KEY" ]; then
  echo "Both issuer.crt and issuer.key exist."
else
  step certificate create identity.linkerd.cluster.local issuer.crt issuer.key \
    --profile=intermediate-ca --not-after=8760h --no-password --insecure \
    --ca=ca.crt --ca-key=ca.key
fi

# Set expiry date one year from now
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo "Linux"
  exp=$(date -d '+8760 hour' +"%Y-%m-%dT%H:%M:%SZ")
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "macOS"
  exp=$(date -v+8760H +"%Y-%m-%dT%H:%M:%SZ")
else
  echo "System $OSTYPE is not supported."
  exit 1;
fi

helm install hm-linkerd \
  --set-file identityTrustAnchorsPEM=ca.crt \
  --set-file identity.issuer.tls.crtPEM=issuer.crt \
  --set-file identity.issuer.tls.keyPEM=issuer.key \
  --set identity.issuer.crtExpiry="$exp" \
  linkerd/linkerd2
linkerd check

helm install hm-linkerd-viz linkerd/linkerd-viz
linkerd viz check

helm install hm-linkerd-jaeger linkerd/linkerd-jaeger
linkerd jaeger check


# 3. Install hm-chart
helm install hm-chart hm-chart
linkerd check
