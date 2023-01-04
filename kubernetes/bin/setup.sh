#!/usr/bin/env bash
set -e


is_debug=false
# source bin/utils/focus_opa_debug.sh

export ORG_DOMAIN="k8s-hongbomiao.com"

if [ $is_debug = false ]; then
  CLUSTERS=("west" "east")
else
  CLUSTERS=("west")
fi

source bin/utils/setup_k3d.sh "${CLUSTERS[@]}"

if [ $is_debug = false ]; then
  source bin/utils/install_linkerd.sh "${CLUSTERS[@]}"
fi

source bin/utils/install_ingress.sh "${CLUSTERS[@]}"
if [ $is_debug = false ]; then
  source bin/utils/patch_ingress.sh "${CLUSTERS[@]}"
fi
sleep 30

if [ $is_debug = false ]; then
  source bin/utils/install_linkerd_multicluster.sh "${CLUSTERS[@]}"
fi


echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="


if [ $is_debug = false ]; then
  source bin/utils/install_linkerd_viz.sh
fi

if [ $is_debug = false ]; then
  echo "# List gateways"
  linkerd multicluster gateways
  echo "=================================================="
fi

if [ $is_debug = false ]; then
  source bin/utils/install_linkerd_jaeger.sh
  # source bin/utils/install_linkerd_buoyant.sh

  source bin/utils/install_fluent_bit.sh
  source bin/utils/install_minio.sh
  # source bin/utils/install_yugabyte.sh
  source kubernetes/bin/utils/install_kafka.sh
  source kubernetes/bin/utils/install_debezium.sh
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
