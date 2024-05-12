#!/usr/bin/env bash
set -e

is_debug=false
# source kubernetes/bin/utils/focus_opa_debug.sh

export ORG_DOMAIN="k8s-hongbomiao.com"

if [ $is_debug = false ]; then
  CLUSTERS=("west" "east")
else
  CLUSTERS=("west")
fi

source kubernetes/bin/utils/set_up_k3d.sh "${CLUSTERS[@]}"

if [ $is_debug = false ]; then
  source kubernetes/bin/utils/install_linkerd.sh "${CLUSTERS[@]}"
fi

for cluster in "${CLUSTERS[@]}"; do
  echo "# Install Ingress on: k3d-${cluster}"
  helm upgrade \
    ingress-nginx \
    ingress-nginx \
    --install \
    --repo=https://kubernetes.github.io/ingress-nginx \
    --kube-context="k3d-${cluster}" \
    --namespace=ingress-nginx \
    --create-namespace
  echo "=================================================="
done
if [ $is_debug = false ]; then
  source kubernetes/bin/utils/patch_ingress.sh "${CLUSTERS[@]}"
fi
sleep 30

if [ $is_debug = false ]; then
  source kubernetes/bin/utils/install_linkerd_multicluster.sh "${CLUSTERS[@]}"
fi


echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="


if [ $is_debug = false ]; then
  source kubernetes/bin/utils/install_linkerd_viz.sh
fi

if [ $is_debug = false ]; then
  echo "# List gateways"
  linkerd multicluster gateways
  echo "=================================================="
fi

if [ $is_debug = false ]; then
  source kubernetes/bin/utils/install_linkerd_jaeger.sh
  # source kubernetes/bin/utils/install_linkerd_buoyant.sh

  source kubernetes/bin/utils/install_fluent_bit.sh
  source kubernetes/bin/utils/install_minio.sh
  # source kubernetes/bin/utils/install_yugabyte.sh
  source kubernetes/bin/utils/install_hm_kafka.sh
  source kubernetes/bin/utils/install_hm_kafka_opa_kafka_connect.sh
fi

source kubernetes/bin/utils/install_elastic.sh
source kubernetes/bin/utils/install_argocd.sh
source kubernetes/bin/utils/install_dgraph.sh
source kubernetes/bin/utils/install_postgres.sh
source kubernetes/bin/utils/install_west_cluster_apps.sh
source kubernetes/bin/utils/secure_opal_server.sh
source kubernetes/bin/utils/secure_config_loader.sh

if [ $is_debug = false ]; then
  sleep 120

  echo "# East cluster"
  kubectl config use-context k3d-east
  echo "=================================================="

  source kubernetes/bin/utils/install_east_cluster_apps.sh
  # source kubernetes/bin/utils/verify_linkerd_multicluster.sh

  echo "# West cluster"
  kubectl config use-context k3d-west
  echo "=================================================="

  # source kubernetes/bin/utils/install_hm_streaming.sh
fi
