#!/usr/bin/env bash
set -e

# Elastic
# Install custom resource definitions and the Elasticsearch operator with its RBAC rules
# https://www.elastic.co/guide/en/cloud-on-k8s/current/k8s-deploy-eck.html
echo "# Install custom resource definitions and the Elasticsearch operator with its RBAC rules"
kubectl apply --filename=https://download.elastic.co/downloads/eck/2.7.0/crds.yaml
kubectl apply --filename=https://download.elastic.co/downloads/eck/2.7.0/operator.yaml
# kubectl delete --filename=https://download.elastic.co/downloads/eck/2.7.0/crds.yaml
# kubectl delete --filename=https://download.elastic.co/downloads/eck/2.7.0/operator.yaml
sleep 30
echo "=================================================="

# Monitor the Elasticsearch operator logs
# kubectl logs --namespace=hm-elastic-system statefulset.apps/elastic-operator --follow

# Deploy Elasticsearch
echo "# Deploy Elasticsearch, Kibana, AMP"
kubectl apply --filename=kubernetes/manifests/elastic/hm-elastic-namespace.yaml
kubectl apply --filename=kubernetes/manifests/elastic/hm-elasticsearch.yaml
kubectl apply --filename=kubernetes/manifests/elastic/hm-kibana.yaml
kubectl apply --filename=kubernetes/manifests/elastic/hm-apm.yaml
# kubectl delete --filename=kubernetes/manifests/elastic/hm-apm.yaml
# kubectl delete --filename=kubernetes/manifests/elastic/hm-kibana.yaml
# kubectl delete --filename=kubernetes/manifests/elastic/hm-elasticsearch.yaml
# kubectl delete --filename=kubernetes/manifests/elastic/hm-elastic-namespace.yaml
sleep 60
echo "=================================================="

echo "# Check Elastic"
for d in hm-apm-apm-server hm-kibana-kb; do
  kubectl rollout status deployment/${d} --namespace=hm-elastic
done
echo "=================================================="

# Elasticsearch
# kubectl port-forward service/hm-elasticsearch-es-http --namespace=hm-elastic 9200:9200
# ELASTIC_PASSWORD=$(kubectl get secret hm-elasticsearch-es-elastic-user \
#   --namespace=hm-elastic \
#   --output=go-template="{{.data.elastic | base64decode}}")
# curl --user "elastic:${ELASTIC_PASSWORD}" --insecure "https://localhost:9200"

# Kibana
# kubectl port-forward service/hm-kb-http --namespace=hm-elastic 5601:5601
# Username: elastic
# Password:
# kubectl get secret hm-elasticsearch-es-elastic-user \
#   --namespace=hm-elastic \
#   --output=jsonpath="{.data.elastic}" | base64 --decode; echo

# Elastic APM
echo "# Save Elastic APM tls.crt locally"
rm -f kubernetes/data/elastic-apm/tls.crt
kubectl get secret hm-apm-apm-http-certs-public \
  --namespace=hm-elastic \
  --output=go-template='{{index .data "tls.crt" | base64decode }}' \
  > kubernetes/data/elastic-apm/tls.crt
echo "=================================================="

echo "# Save Elastic APM token in Kubernetes secret"
kubectl apply --filename=kubernetes/manifests/west/hm-namespace.yaml
ELASTIC_APM_TOKEN=$(kubectl get secret hm-apm-apm-token \
  --namespace=hm-elastic \
  --output=go-template='{{index .data "secret-token" | base64decode}}')
kubectl create secret generic hm-elastic-apm \
  --namespace=hm \
  --from-literal="token=${ELASTIC_APM_TOKEN}"
echo "=================================================="
