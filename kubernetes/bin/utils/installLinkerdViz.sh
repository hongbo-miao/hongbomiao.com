#!/usr/bin/env bash

set -e


# Install Linkerd viz
echo "# Install Linkerd viz"
linkerd viz install \
  --set=jaegerUrl=jaeger.linkerd-jaeger:16686,clusterDomain="west.${ORG_DOMAIN}" | \
  kubectl apply --filename=-
echo "=================================================="
# linkerd viz dashboard --context=k3d-west
sleep 60


# Check Linkerd viz
echo "# Check Linkerd viz"
linkerd viz check
echo "=================================================="
