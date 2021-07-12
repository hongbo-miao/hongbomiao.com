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
kind create cluster --name=west --config=kubernetes/kind/west-cluster-config.yaml
kind create cluster --name=east --config=kubernetes/kind/east-cluster-config.yaml
# kind delete cluster --name=west
# kind delete cluster --name=east

# Install Linkerd
linkerd install \
  --identity-trust-anchors-file=kubernetes/certificates/root.crt \
  --identity-issuer-certificate-file=kubernetes/certificates/issuer.crt \
  --identity-issuer-key-file=kubernetes/certificates/issuer.key |
  tee \
    >(kubectl --context=kind-west apply --filename=-) \
    >(kubectl --context=kind-east apply --filename=-)

# Install Ingress
for ctx in kind-west kind-east; do
  echo "Installing Ingress on ${ctx}"
  VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl apply --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${VERSION}/deploy/static/provider/kind/deploy.yaml"
  echo "-------------"
done

# Patch Ingress
for ctx in kind-west kind-east; do
  echo "Patching Ingress on ${ctx}"
  kubectl patch configmap ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patch/ingress-nginx-controller-configmap-patch.yaml)"
  kubectl patch deployment ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patch/ingress-nginx-controller-deployment-patch.yaml)"
  echo "-------------"
done

# Check Linkerd
for ctx in kind-west kind-east; do
  echo "Checking cluster: ${ctx}"
  linkerd --context=${ctx} check || break
  echo "-------------"
done

# Install Linkerd multicluster
for ctx in kind-west kind-east; do
  echo "Installing on cluster: ${ctx}"
  linkerd --context=${ctx} multicluster install |
    kubectl --context=${ctx} apply -f - || break
  # linkerd multicluster uninstall | kubectl delete -f -
  echo "-------------"
done

# Check Linkerd multicluster
for ctx in kind-west kind-east; do
  echo "Checking gateway on cluster: ${ctx}"
  kubectl --context=${ctx} -n linkerd-multicluster \
    rollout status deploy/linkerd-gateway || break
  echo "-------------"
done

# Check load balancer
for ctx in kind-west kind-east; do
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
linkerd --context=kind-east multicluster link --cluster-name=kind-east |
  kubectl --context=kind-west apply -f -
