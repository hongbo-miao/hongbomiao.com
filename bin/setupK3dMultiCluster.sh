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
  echo "-------------"
done

# Create clusters
k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
k3d cluster create east --config=kubernetes/k3d/east-cluster-config.yaml
#	k3d cluster delete west
#	k3d cluster delete east

kubectl config use-context k3d-west
# kubectl config use-context k3d-east


# Install Linkerd
for cluster in west east; do
  # Wait the cluster is ready to install Linkerd
  while ! linkerd check --context="k3d-${cluster}" --pre ; do :; done

  domain="${cluster}.${ORG_DOMAIN}"
  crt="${CA_DIR}/${cluster}-issuer.crt"
  key="${CA_DIR}/${cluster}-issuer.key"

  echo "Install Linkerd on k3d-${cluster}"
  linkerd install \
    --context="k3d-${cluster}" \
    --cluster-domain=${domain} \
    --identity-trust-domain=${domain} \
    --identity-trust-anchors-file="${CA_DIR}/ca.crt" \
    --identity-issuer-certificate-file=${crt} \
    --identity-issuer-key-file=${key} \
    --disable-heartbeat | \
    kubectl apply --context="k3d-${cluster}" --filename=-
  echo "-------------"
done


# Install Ingress
for cluster in west east; do
  # Wait the Linkerd is ready
  while ! linkerd check --context="k3d-${cluster}" ; do :; done

  echo "Install Ingress on k3d-${cluster}"
  VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl apply --context="k3d-${cluster}" --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/cloud/deploy.yaml"
  # Local: kubectl apply --context="k3d-${cluster}" --filename=kubernetes/manifests/ingress-nginx.yaml
  echo "-------------"

  echo "Patch Ingress on k3d-${cluster}"
  kubectl patch configmap ingress-nginx-controller --context="k3d-${cluster}" --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-configmap-patch.yaml)"
  kubectl patch deployment ingress-nginx-controller --context="k3d-${cluster}" --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-deployment-patch.yaml)"
  echo "-------------"
done


# Install Linkerd multicluster
for cluster in west east; do
  # Wait the Linkerd is ready
  while ! linkerd check --context="k3d-${cluster}" ; do :; done

  echo "Install on cluster: k3d-${cluster}"
  linkerd multicluster install --context="k3d-${cluster}" | kubectl apply --context="k3d-${cluster}" --filename=-
  # linkerd multicluster uninstall | kubectl delete --filename=-
  echo "-------------"
done


# Check Linkerd multicluster
for cluster in west east; do
  echo "Check gateway on cluster: k3d-${cluster}"
  kubectl rollout status deploy/linkerd-gateway --context="k3d-${cluster}" --namespace=linkerd-multicluster
  echo "-------------"
done


# Check load balancer
for cluster in west east; do
  echo "Check load balancer on cluster: k3d-${cluster}"
  while [ "$(kubectl get service --context="k3d-${cluster}" --namespace=linkerd-multicluster \
    --output='custom-columns=:.status.loadBalancer.ingress[0].ip' \
    --no-headers)" = "<none>" ]; do
    printf '.'
    sleep 1
  done
  echo "-------------"
done


# Link the cluster
EAST_IP=$(kubectl get service --context="k3d-east" --namespace=ingress-nginx ingress-nginx-controller --output='go-template={{ (index .status.loadBalancer.ingress 0).ip }}')
linkerd multicluster link \
  --context=k3d-east \
  --cluster-name=k3d-east \
  --api-server-address="https://${EAST_IP}:6443" | \
  kubectl apply --context=k3d-west --filename=-

WEST_IP=$(kubectl get service --context="k3d-west" --namespace=ingress-nginx ingress-nginx-controller --output='go-template={{ (index .status.loadBalancer.ingress 0).ip }}')
linkerd link \
  --context=k3d-west multicluster \
  --cluster-name=k3d-west \
  --api-server-address="https://${WEST_IP}:6443" | \
  kubectl apply --context=k3d-east --filename=-


# Check multicluster
linkerd multicluster check --context=k3d-west
linkerd multicluster check --context=k3d-east


# Install Linkerd Viz
for cluster in west east; do
  echo "Install Linkerd Viz on k3d-${cluster}"
  kubectl config use-context "k3d-${cluster}"

  domain="${cluster}.${ORG_DOMAIN}"
  linkerd viz install \
    --context="k3d-${cluster}" \
    --set=jaegerUrl=jaeger.linkerd-jaeger:16686,clusterDomain=${domain} | \
    kubectl apply --filename=-
  echo "-------------"
done

# linkerd viz dashboard --context=k3d-west
# linkerd viz dashboard --context=k3d-east

# List gateways
linkerd --context=k3d-west multicluster gateways
linkerd --context=k3d-east multicluster gateways
