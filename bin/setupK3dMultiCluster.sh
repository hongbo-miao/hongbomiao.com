#!/usr/bin/env bash

set -e

# Generate certificates
cd kubernetes/certificates
step certificate create root.linkerd.cluster.local root.crt root.key --profile=root-ca --no-password --insecure
step certificate create identity.linkerd.cluster.local issuer.crt issuer.key \
  --profile=intermediate-ca --not-after=8760h --no-password --insecure \
  --ca=root.crt --ca-key=root.key

# Create clusters
cd ../..
k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
k3d cluster create east --config=kubernetes/k3d/east-cluster-config.yaml
#	k3d cluster delete west
#	k3d cluster delete east

# Install Linkerd
linkerd install \
  --identity-trust-anchors-file=kubernetes/certificates/root.crt \
  --identity-issuer-certificate-file=kubernetes/certificates/issuer.crt \
  --identity-issuer-key-file=kubernetes/certificates/issuer.key |
  tee \
    >(kubectl --context=k3d-west apply --filename=-) \
    >(kubectl --context=k3d-east apply --filename=-)

# Install Ingress
for ctx in k3d-west k3d-east; do
  echo "Installing Ingress on ${ctx}"
  VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl apply --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/cloud/deploy.yaml"
  # Local: kubectl apply --filename=kubernetes/manifests/ingress-nginx.yaml
  echo "-------------"
done

# Patch Ingress
for ctx in k3d-west k3d-east; do
  echo "Patching Ingress on ${ctx}"
  kubectl patch configmap ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-configmap-patch.yaml)"
  kubectl patch deployment ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-deployment-patch.yaml)"
  echo "-------------"
done

# Check Linkerd
for ctx in k3d-west k3d-east; do
  echo "Checking cluster: ${ctx}"
  linkerd --context=${ctx} check || break
  echo "-------------"
done

# Install Linkerd multicluster
for ctx in k3d-west k3d-east; do
  echo "Installing on cluster: ${ctx}"
  linkerd --context=${ctx} multicluster install |
    kubectl --context=${ctx} apply --filename=- || break
  # linkerd multicluster uninstall | kubectl delete --filename=-
  echo "-------------"
done

# Check Linkerd multicluster
for ctx in k3d-west k3d-east; do
  echo "Checking gateway on cluster: ${ctx}"
  kubectl --context=${ctx} -n linkerd-multicluster \
    rollout status deploy/linkerd-gateway || break
  echo "-------------"
done

# Check load balancer
for ctx in k3d-west k3d-east; do
  echo "Checking load balancer on cluster: ${ctx}"
  while [ "$(kubectl --context=${ctx} -n linkerd-multicluster get service \
    -o 'custom-columns=:.status.loadBalancer.ingress[0].ip' \
    --no-headers)" = "<none>" ]; do
    printf '.'
    sleep 1
  done
  echo "-------------"
done

# Link the cluster
linkerd --context=k3d-east multicluster link --cluster-name=k3d-east |
  kubectl --context=k3d-west apply --filename=-
