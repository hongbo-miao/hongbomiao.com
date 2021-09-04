#!/usr/bin/env bash

set -e


# Clean
echo "# Clean"
rm -f kubernetes/certificates/ca.crt
rm -f kubernetes/certificates/ca.key
rm -f kubernetes/certificates/east-issuer.crt
rm -f kubernetes/certificates/east-issuer.key
rm -f kubernetes/certificates/west-issuer.crt
rm -f kubernetes/certificates/west-issuer.key
rm -f kubernetes/data/elastic-apm/tls.crt
echo "=================================================="


# Generate OPAL SSH key
ssh-keygen -t rsa -b 4096 -m pem
# /Users/homiao/Clouds/Git/hongbomiao.com/kubernetes/data/opal-server/id_rsa


# Generate certificates
echo "# Generate certificates"
ORG_DOMAIN="k8s-hongbomiao.com"
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
done
echo "=================================================="


# Create clusters
echo "# Create clusters"
k3d cluster create west --config=kubernetes/k3d/west-cluster-config.yaml
k3d cluster create east --config=kubernetes/k3d/east-cluster-config.yaml

# k3d cluster delete west
# k3d cluster delete east

# kubectl config use-context k3d-west
# kubectl config use-context k3d-east

sleep 30
echo "=================================================="


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
  INGRESS_VERSION=$(curl https://raw.githubusercontent.com/kubernetes/ingress-nginx/master/stable.txt)
  kubectl apply \
    --context="k3d-${cluster}" \
    --filename="https://raw.githubusercontent.com/kubernetes/ingress-nginx/${INGRESS_VERSION}/deploy/static/provider/cloud/deploy.yaml"
  # Local: kubectl apply --context="k3d-${cluster}" --filename=kubernetes/manifests/ingress-nginx.yaml
  echo "=================================================="

  echo "# Patch Ingress on: k3d-${cluster}"
  kubectl patch configmap ingress-nginx-controller \
    --context="k3d-${cluster}" \
    --namespace=ingress-nginx \
    --patch "$(cat kubernetes/patches/ingress-nginx-controller-configmap-patch.yaml)"
  kubectl patch deployment ingress-nginx-controller \
    --context="k3d-${cluster}" \
    --namespace=ingress-nginx \
    --patch "$(cat kubernetes/patches/ingress-nginx-controller-deployment-patch.yaml)"
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
  domain="${cluster}.${ORG_DOMAIN}"
  linkerd multicluster install \
    --context="k3d-${cluster}" \
    --set=identityTrustDomain=${domain} | \
    kubectl apply --context="k3d-${cluster}" --filename=-
  # linkerd multicluster uninstall | kubectl delete --filename=-
  echo "=================================================="
done
sleep 30


# Check Linkerd gateway
for cluster in west east; do
  echo "# Check gateway on: k3d-${cluster}"
  kubectl rollout status deploy/linkerd-gateway \
    --context="k3d-${cluster}" \
    --namespace=linkerd-multicluster
  echo "=================================================="
done


# Check load balancer
for cluster in west east; do
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


# Link the cluster
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
echo "=================================================="
sleep 30


# Check Linkerd multicluster
for cluster in west east; do
  echo "# Check Linkerd multicluster on: k3d-${cluster}"
  # Add `continue` here because of https://github.com/olix0r/l2-k3d-multi/issues/5
  linkerd multicluster check --context="k3d-${cluster}" || continue
  echo "=================================================="
done


# West cluster
echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="


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


# List gateways
echo "# List gateways"
linkerd multicluster gateways
echo "=================================================="


# Install Linkerd Jaeger
echo "# Install Linkerd Jaeger"
linkerd jaeger install | \
  kubectl apply --filename=-
echo "=================================================="
sleep 30

# linkerd jaeger dashboard --context=k3d-west


# Check Linkerd Jaeger
echo "# Check Linkerd Jaeger"
linkerd jaeger check
echo "=================================================="


# Install Linkerd buoyant
# echo "# Install Linkerd buoyant"
# curl -sL buoyant.cloud/install | sh
# linkerd buoyant install | \
#   kubectl apply --filename=-
# echo "=================================================="
# sleep 30


# Elastic
# Install custom resource definitions and the Elasticsearch operator with its RBAC rules
echo "# Install custom resource definitions and the Elasticsearch operator with its RBAC rules"
kubectl apply --filename=https://download.elastic.co/downloads/eck/1.7.0/crds.yaml
kubectl apply --filename=https://download.elastic.co/downloads/eck/1.7.0/operator.yaml
echo "=================================================="
sleep 30


# Monitor the Elasticsearch operator logs
# kubectl logs --namespace=elastic-system --filename=statefulset.apps/elastic-operator


# Deploy Elasticsearch
echo "# Deploy Elasticsearch, Kibana, AMP"
kubectl apply --filename=kubernetes/config/elastic
# Delete: kubectl delete --filename=kubernetes/config/elastic
echo "=================================================="
sleep 60


echo "# Check Elastic"
for d in hm-apm-apm-server hm-kibana-kb; do
  kubectl --namespace=elastic rollout status deployment/${d}
done
echo "=================================================="


# Elasticsearch
# kubectl port-forward service/hm-elasticsearch-es-http --namespace=elastic 9200:9200
# ELASTIC_PASSWORD=$(kubectl get secret hm-elasticsearch-es-elastic-user \
#   --namespace=elastic \
#   --output=go-template="{{.data.elastic | base64decode}}")
# curl -u "elastic:${ELASTIC_PASSWORD}" -k "https://localhost:9200"

# Kibana
# kubectl port-forward service/hm-kb-http --namespace=elastic 5601:5601
# Username: elastic
# Password:
# kubectl get secret hm-elasticsearch-es-elastic-user \
#   --namespace=elastic \
#   --output=jsonpath="{.data.elastic}" | base64 --decode; echo

# Elastic APM
echo "# Save Elastic APM tls.crt locally"
kubectl get secret hm-apm-apm-http-certs-public \
  --namespace=elastic \
  --output=go-template='{{index .data "tls.crt" | base64decode }}' \
  > kubernetes/data/elastic-apm/tls.crt
echo "=================================================="

echo "# Save Elastic APM token in Kubernetes secret"
kubectl create namespace hm
ELASTIC_APM_TOKEN=$(kubectl get secret hm-apm-apm-token \
  --namespace=elastic \
  --output=go-template='{{index .data "secret-token" | base64decode}}')
kubectl create secret generic hm-elastic-apm \
  --namespace=hm \
  --from-literal="token=${ELASTIC_APM_TOKEN}"
echo "=================================================="


# Install Argo CD
echo "# Install Argo CD"
kubectl create namespace argocd
kubectl apply \
  --namespace=argocd \
  --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
# Local:
# kubectl apply \
#   --namespace=argocd \
#   --filename=kubernetes/manifests/argocd.yaml
# Delete:
# kubectl delete \
#   --namespace=argocd \
#   --filename=https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
echo "=================================================="
sleep 30


echo "# Check Argo CD"
for d in dex-server redis repo-server server; do
  kubectl rollout status deployment/argocd-${d} --namespace=argocd
done
echo "=================================================="


# Install the app by Argo CD
echo "# Install the app"
kubectl port-forward service/argocd-server --namespace=argocd 31026:443 &
ARGOCD_PASSWORD=$(kubectl get secret argocd-initial-admin-secret \
  --namespace=argocd \
  --output=jsonpath="{.data.password}" | \
  base64 -d && echo)
argocd login localhost:31026 --username=admin --password="${ARGOCD_PASSWORD}" --insecure
kubectl apply --filename=kubernetes/config/argocd/hm-application.yaml
argocd app sync hm-application --local=kubernetes/config/west
# Delete: argocd app delete hm-application --yes
echo "=================================================="

echo "# Create OPAL server secret"
OPAL_AUTH_MASTER_TOKEN=IWjW0bYcTIfm6Y5JNjp4DdgopC6rYSxT4yrPbtLiTU0
kubectl create secret generic hm-opal-server-secret \
  --namespace=hm-opa \
  --from-file=id_rsa=kubernetes/data/opal-server/id_rsa \
  --from-file=id_rsa.pub=kubernetes/data/opal-server/id_rsa.pub \
  --from-literal=opal_auth_master_token="${OPAL_AUTH_MASTER_TOKEN}"
echo "=================================================="

echo "# Check OPAL server"
kubectl rollout status deployment/opal-server-deployment --namespace=hm-opa
kubectl port-forward service/opal-server-service --namespace=hm-opa 7002:7002 &
echo "=================================================="

echo "# Create OPAL client secret"
OPAL_CLIENT_TOKEN=$(opal-client obtain-token "${OPAL_AUTH_MASTER_TOKEN}" \
  --server-url=http://localhost:7002)
kubectl create secret generic hm-opal-client-secret \
  --namespace=hm \
  --from-literal=opal_client_token="${OPAL_CLIENT_TOKEN}"
echo "=================================================="
sleep 120


# East cluster
echo "# East cluster"
kubectl config use-context k3d-east
echo "=================================================="


# Install the app by Argo CD
echo "# Install the app"
kubectl apply --filename=kubernetes/config/east/hm-namespace.yaml
kubectl apply --filename=kubernetes/config/east
# Delete: kubectl delete --filename=kubernetes/config/east
echo "=================================================="
sleep 30

# Verify Linkerd multicluster setup correctly by checking the endpoints in west
# and verify that they match the gateway’s public IP address in east
# kubectl get endpoint grpc-server-service-k3d-east \
#   --context=k3d-west \
#   --namespace=hm \
#   --output="custom-columns=ENDPOINT_IP:.subsets[*].addresses[*].ip"
# kubectl linkerd-multicluster get service linkerd-gateway \
#   --context=k3d-east \
#   --namespace=hm \
#   --output="custom-columns=GATEWAY_IP:.status.loadBalancer.ingress[*].ip"


# West cluster
echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="


# echo "# Install hm-streaming"
# flink run-application \
#   --target kubernetes-application \
#   -Dkubernetes.namespace=hm-flink \
#   -Dkubernetes.cluster-id=hm-flink-cluster \
#   -Dkubernetes.container.image=hongbomiao/hm-streaming:latest \
#   -Dkubernetes.container.image.pull-policy=Always \
#   -Dkubernetes.jobmanager.service-account=flink-serviceaccount \
#   local:///opt/flink/usrlib/streaming-0.1.jar
# Delete: kubectl delete deployment/hm-flink-cluster --namespace=hm-flink
# echo "=================================================="
