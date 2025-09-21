#!/usr/bin/env bash
set -e

echo "# Install Linkerd viz"
linkerd viz install \
  --set=jaegerUrl=jaeger.linkerd-jaeger:16686,clusterDomain="west.${ORG_DOMAIN}" | \
  kubectl apply --filename=-
# linkerd viz dashboard --context=k3d-west
sleep 60
echo "=================================================="

# Check Linkerd viz
echo "# Check Linkerd viz"
linkerd viz check
echo "=================================================="
