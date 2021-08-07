#!/usr/bin/env bash

set -e


# Install Linkerd
linkerd check --pre

echo "Install Linkerd"
linkerd install --disable-heartbeat | kubectl apply --filename=-
# Production: linkerd install --disable-heartbeat --ha | kubectl apply --filename=-
linkerd check

echo "Install Linkerd Viz"
linkerd viz install --set=jaegerUrl=jaeger.linkerd-jaeger:16686 | kubectl apply --filename=-
linkerd viz check

echo "Install Linkerd Jaeger"
linkerd jaeger install | kubectl apply --filename=-
linkerd jaeger check


# Patch Ingress
kubectl patch configmap ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-configmap-patch.yaml)"
kubectl patch deployment ingress-nginx-controller --namespace=ingress-nginx --patch "$(cat kubernetes/patches/ingress-nginx-controller-deployment-patch.yaml)"


# Install Argo CD
echo "Install Argo CD"
kubectl create namespace argocd
kubectl apply --namespace=argocd --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
# Local: kubectl apply --namespace=argocd --filename=kubernetes/manifests/argocd.yaml

for deploy in "dex-server" "redis" "repo-server" "server"; do
  kubectl --namespace=argocd rollout status deploy/argocd-${deploy}
done


# Install the app by Argo CD
echo "Install the app"
kubectl port-forward service/argocd-server --namespace=argocd 31026:443 &
ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d && echo)
argocd login localhost:31026 --username=admin --password="${ARGOCD_PASSWORD}" --insecure
kubectl apply --filename=kubernetes/config/argocd/hm-application.yaml
argocd app sync hm-application --local=kubernetes/config


# Install the app by Kubernetes files
# echo "Install the app"
# kubectl apply --filename=kubernetes/config/hm-namespace.yaml
# kubectl apply --filename=kubernetes/config


# Install Dgraph (https://dgraph.io/docs/deploy/kubernetes)
# linkerd inject https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-ha/dgraph-ha.yaml | kubectl apply --namespace=hm --filename=-
# Delete: kubectl delete --namespace=hm --filename=https://raw.githubusercontent.com/dgraph-io/dgraph/master/contrib/config/kubernetes/dgraph-ha/dgraph-ha.yaml
#         kubectl delete persistentvolumeclaims --namespace=hm --selector=app=dgraph
# Local: kubectl apply --namespace=hm --filename=kubernetes/manifests/dgraph-ha.yaml
