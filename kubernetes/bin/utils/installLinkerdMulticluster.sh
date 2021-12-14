#!/usr/bin/env bash
set -e

CLUSTERS=("$@")

for cluster in "${CLUSTERS[@]}"; do
  echo "# Install Linkerd multicluster on: k3d-${cluster}"
  domain="${cluster}.${ORG_DOMAIN}"
  linkerd multicluster install \
    --context="k3d-${cluster}" \
    --set=identityTrustDomain="${domain}" | \
    kubectl apply --context="k3d-${cluster}" --filename=-
  # linkerd multicluster uninstall | kubectl delete --filename=-
  echo "=================================================="
done
sleep 30

for cluster in "${CLUSTERS[@]}"; do
  echo "# Check gateway on: k3d-${cluster}"
  kubectl rollout status deploy/linkerd-gateway \
    --context="k3d-${cluster}" \
    --namespace=linkerd-multicluster
  echo "=================================================="
done

for cluster in "${CLUSTERS[@]}"; do
  echo "# Check load balancer on: k3d-${cluster}"
  while [ "$(kubectl get service \
    --context="k3d-${cluster}" \
    --namespace=linkerd-multicluster \
    --output="custom-columns=:.status.loadBalancer.ingress[0].ip" \
    --no-headers)" = "<none>" ]; do
    printf "."
    sleep 1
  done
  echo "=================================================="
done

echo "# Link the cluster"
# https://github.com/olix0r/l2-k3d-multi/blob/4cb5e6994671a5f6f507b7ad281b8c008927c9d0/link.sh#L12-L22
# Unfortunately, the credentials have the API server IP as addressed from localhost and not the docker network, so we have to patch that up.
EAST_IP=$(kubectl get service \
  --context="k3d-east" \
  --namespace=ingress-nginx ingress-nginx-controller \
  --output="go-template={{ (index .status.loadBalancer.ingress 0).ip }}")
linkerd multicluster link \
  --cluster-name=k3d-east \
  --api-server-address="https://${EAST_IP}:6443" | \
    kubectl apply --context=k3d-west --filename=-

WEST_IP=$(kubectl get service \
  --context="k3d-west" \
  --namespace=ingress-nginx ingress-nginx-controller \
  --output="go-template={{ (index .status.loadBalancer.ingress 0).ip }}")
linkerd multicluster link \
  --cluster-name=k3d-west \
  --api-server-address="https://${WEST_IP}:6443" | \
    kubectl apply --context=k3d-east --filename=-
sleep 30
echo "=================================================="

for cluster in "${CLUSTERS[@]}"; do
  echo "# Check Linkerd multicluster on: k3d-${cluster}"
  # Add `continue` here because of https://github.com/olix0r/l2-k3d-multi/issues/5
  linkerd multicluster check --context="k3d-${cluster}" || continue
  echo "=================================================="
done
