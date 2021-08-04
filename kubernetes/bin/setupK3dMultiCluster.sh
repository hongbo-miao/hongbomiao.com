#!/usr/bin/env bash

set -e


# Generate certificates
ORG_DOMAIN="k3d.hongbomiao.com"
CA_DIR="kubernetes/certificates"

step certificate create "identity.linkerd.${ORG_DOMAIN}" \
  "${CA_DIR}/ca.crt" "${CA_DIR}/ca.key" \
  --profile=root-ca \
  --no-password \
  --insecure

for cluster in west east; do
  domain="${cluster}.${ORG_DOMAIN}"
  crt="${CA_DIR}/${cluster}-issuer.crt"
  key="${CA_DIR}/${cluster}-issuer.key"

  step certificate create "identity.linkerd.${domain}" ${crt} ${key} \
    --ca="${CA_DIR}/ca.crt" \
    --ca-key="${CA_DIR}/ca.key" \
    --profile=intermediate-ca \
    --not-after=8760h \
    --no-password \
    --insecure
  echo "=================================================="
done

# Create clusters
k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
k3d cluster create east --config=kubernetes/k3d/east-cluster-config.yaml

sleep 30

#	k3d cluster delete west
#	k3d cluster delete east

# kubectl config use-context k3d-west
# kubectl config use-context k3d-east


# Check Linkerd installation environment
for cluster in west east; do
  echo "# Check Linkerd installation environment on: k3d-${cluster}"
  while ! linkerd check --context="k3d-${cluster}" --pre ; do :; done
  echo "=================================================="
done


# Install Linkerd
for cluster in west east; do
  domain="${cluster}.${ORG_DOMAIN}"
  crt="${CA_DIR}/${cluster}-issuer.crt"
  key="${CA_DIR}/${cluster}-issuer.key"

  echo "# Install Linkerd on: k3d-${cluster}"
  linkerd install \
    --cluster-domain=${domain} \
    --identity-trust-domain=${domain} \
    --identity-trust-anchors-file="${CA_DIR}/ca.crt" \
    --identity-issuer-certificate-file=${crt} \
    --identity-issuer-key-file=${key} \
    --disable-heartbeat | \
    kubectl apply --context="k3d-${cluster}" --filename=-
  echo "=================================================="
done

sleep 30


# Check Linkerd
for cluster in west east; do
  echo "# Check Linkerd on: k3d-${cluster}"
  while ! linkerd check --context="k3d-${cluster}" ; do :; done
  echo "=================================================="
done


# Install Ingress with patch
for cluster in west east; do
  echo "# Install Ingress on: k3d-${cluster}"
  VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl apply --context="k3d-${cluster}" --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/cloud/deploy.yaml"
  # Local: kubectl apply --context="k3d-${cluster}" --filename=kubernetes/manifests/ingress-nginx.yaml
  echo "=================================================="

  echo "# Patch Ingress on: k3d-${cluster}"
  kubectl patch configmap ingress-nginx-controller --context="k3d-${cluster}" --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-configmap-patch.yaml)"
  kubectl patch deployment ingress-nginx-controller --context="k3d-${cluster}" --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-deployment-patch.yaml)"
  echo "=================================================="
done

sleep 30


# Check Linkerd
for cluster in west east; do
  echo "# Check Linkerd on: k3d-${cluster}"
  while ! linkerd check --context="k3d-${cluster}" ; do :; done
  echo "=================================================="
done


# Install Linkerd multicluster
for cluster in west east; do
  echo "# Install Linkerd multicluster on: k3d-${cluster}"
  linkerd multicluster install --context="k3d-${cluster}" | kubectl apply --context="k3d-${cluster}" --filename=-
  # linkerd multicluster uninstall | kubectl delete --filename=-
  echo "=================================================="
done

sleep 30


# Check gateway
for cluster in west east; do
  echo "# Check gateway on: k3d-${cluster}"
  kubectl rollout status deploy/linkerd-gateway --context="k3d-${cluster}" --namespace=linkerd-multicluster
  echo "=================================================="
done


# Check load balancer
for cluster in west east; do
  echo "# Check load balancer on: k3d-${cluster}"
  while [ "$(kubectl get service \
    --context="k3d-${cluster}" \
    --namespace=linkerd-multicluster \
    --output='custom-columns=:.status.loadBalancer.ingress[0].ip' \
    --no-headers)" = "<none>" ]; do
    printf '.'
    sleep 1
  done
  echo "=================================================="
done


# Link the cluster
echo "# Link the cluster"
# https://github.com/olix0r/l2-k3d-multi/blob/4cb5e6994671a5f6f507b7ad281b8c008927c9d0/link.sh#L12-L22
# Unfortunately, the credentials have the API server IP as addressed from localhost and not the docker network, so we have to patch that up.
EAST_IP=$(kubectl get service --context="k3d-east" --namespace=ingress-nginx ingress-nginx-controller --output='go-template={{ (index .status.loadBalancer.ingress 0).ip }}')
linkerd multicluster link \
  --cluster-name=k3d-east \
  --api-server-address="https://${EAST_IP}:6443" | \
  kubectl apply --context=k3d-west --filename=-

WEST_IP=$(kubectl get service --context="k3d-west" --namespace=ingress-nginx ingress-nginx-controller --output='go-template={{ (index .status.loadBalancer.ingress 0).ip }}')
linkerd multicluster link \
  --cluster-name=k3d-west \
  --api-server-address="https://${WEST_IP}:6443" | \
  kubectl apply --context=k3d-east --filename=-

sleep 30


# Check Linkerd multicluster
for cluster in west east; do
  echo "# Check Linkerd multicluster on: k3d-${cluster}"
  # Add `continue` here because of https://github.com/olix0r/l2-k3d-multi/issues/5
  linkerd multicluster check --context="k3d-${cluster}" || continue
  echo "=================================================="
done


# Install Linkerd viz
for cluster in west east; do
  echo "# Install Linkerd viz on: k3d-${cluster}"
  domain="${cluster}.${ORG_DOMAIN}"

  linkerd viz install \
    --context="k3d-${cluster}" \
    --set=jaegerUrl=jaeger.linkerd-jaeger:16686,clusterDomain=${domain} | \
    kubectl apply --context="k3d-${cluster}" --filename=-
  echo "=================================================="
done

sleep 30

# linkerd viz dashboard --context=k3d-west
# linkerd viz dashboard --context=k3d-east


# Check Linkerd viz
for cluster in west east; do
  echo "# Check Linkerd viz on: k3d-${cluster}"
  while ! linkerd viz check --context="k3d-${cluster}" ; do :; done
  echo "=================================================="
done


# List gateways
for cluster in west east; do
  # Wait the Linkerd Viz is ready

  echo "# List gateways: k3d-${cluster}"
  linkerd multicluster gateways --context="k3d-${cluster}"
  echo "=================================================="
done
