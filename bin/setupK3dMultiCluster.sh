#!/usr/bin/env bash

set -e


ORG_DOMAIN="k3d.hongbomiao.com"
CA_DIR="kubernetes/certificates"

# Generate certificates
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
  echo "-------------"
done

# Create clusters
k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
k3d cluster create east --config=kubernetes/k3d/east-cluster-config.yaml
#	k3d cluster delete west
#	k3d cluster delete east

kubectl config use-context k3d-west
# kubectl config use-context k3d-east


# Install Ingress
for cluster in west east; do
  echo "Installing Ingress on "k3d-${cluster}""
  VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl --context="k3d-${cluster}" apply --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/cloud/deploy.yaml"
  # Local: kubectl --context="k3d-${cluster}" apply --filename=kubernetes/manifests/ingress-nginx.yaml
  echo "-------------"
done


# Install Linkerd
for cluster in west east; do
  domain="${cluster}.${ORG_DOMAIN}"
  crt="${CA_DIR}/${cluster}-issuer.crt"
  key="${CA_DIR}/${cluster}-issuer.key"

  linkerd --context="k3d-${cluster}" install \
    --cluster-domain=${domain} \
    --identity-trust-domain=${domain} \
    --identity-trust-anchors-file="${CA_DIR}/ca.crt" \
    --identity-issuer-certificate-file=${crt} \
    --identity-issuer-key-file=${key} | \
    kubectl --context="k3d-${cluster}" apply --filename=-
  echo "-------------"
done


# Check Linkerd
for cluster in west east; do
  echo "Checking cluster: "k3d-${cluster}""
  linkerd --context="k3d-${cluster}" check || break
  echo "-------------"
done


# Patch Ingress
for cluster in west east; do
  echo "Patching Ingress on "k3d-${cluster}""
  kubectl --context="k3d-${cluster}" patch configmap ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-configmap-patch.yaml)"
  kubectl --context="k3d-${cluster}" patch deployment ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-deployment-patch.yaml)"
  echo "-------------"
done


# Install Linkerd multicluster
for cluster in west east; do
  echo "Installing on cluster: "k3d-${cluster}""
  linkerd --context="k3d-${cluster}" multicluster install | kubectl --context="k3d-${cluster}" apply --filename=- || break
  # linkerd multicluster uninstall | kubectl delete --filename=-
  echo "-------------"
done


# Check Linkerd multicluster
for cluster in west east; do
  echo "Checking gateway on cluster: "k3d-${cluster}""
  kubectl --context="k3d-${cluster}" --namespace=linkerd-multicluster rollout status deploy/linkerd-gateway || break
  echo "-------------"
done


# Check load balancer
for cluster in west east; do
  echo "Checking load balancer on cluster: "k3d-${cluster}""
  while [ "$(kubectl --context="k3d-${cluster}" --namespace=linkerd-multicluster get service \
    -o 'custom-columns=:.status.loadBalancer.ingress[0].ip' \
    --no-headers)" = "<none>" ]; do
    printf '.'
    sleep 1
  done
  echo "-------------"
done


# Link the cluster
EAST_IP=$(kubectl --context="k3d-east" get service --namespace=ingress-nginx ingress-nginx-controller --output='go-template={{ (index .status.loadBalancer.ingress 0).ip }}')
linkerd --context=k3d-east multicluster link \
  --cluster-name=k3d-east \
  --api-server-address="https://${EAST_IP}:6441" | kubectl --context=k3d-west apply --filename=-


# Check multicluster
linkerd --context=k3d-west multicluster check


# List gateways
linkerd --context=k3d-west multicluster gateways
