#!/usr/bin/env bash
set -e


is_debug=false
# source bin/utils/focusOPADebug.sh

export ORG_DOMAIN="k8s-hongbomiao.com"

if [ $is_debug = false ]; then
  CLUSTERS=("west" "east")
else
  CLUSTERS=("west")
fi

source bin/utils/setupK3d.sh "${CLUSTERS[@]}"

if [ $is_debug = false ]; then
  source bin/utils/installLinkerd.sh "${CLUSTERS[@]}"
fi

source bin/utils/installIngress.sh "${CLUSTERS[@]}"
if [ $is_debug = false ]; then
  source bin/utils/patchIngress.sh "${CLUSTERS[@]}"
fi
sleep 30

if [ $is_debug = false ]; then
  source bin/utils/installLinkerdMulticluster.sh "${CLUSTERS[@]}"
fi


echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="


if [ $is_debug = false ]; then
  source bin/utils/installLinkerdViz.sh
fi

if [ $is_debug = false ]; then
  echo "# List gateways"
  linkerd multicluster gateways
  echo "=================================================="
fi

if [ $is_debug = false ]; then
  source bin/utils/installLinkerdJaeger.sh
  # source bin/utils/installLinkerdBuoyant.sh

  source bin/utils/installFluentBit.sh
  source bin/utils/installMinIO.sh
  # source bin/utils/installYugabyte.sh
  source kubernetes/bin/utils/installKafka.sh
  source kubernetes/bin/utils/installDebezium.sh
fi

source kubernetes/bin/utils/installElastic.sh
source kubernetes/bin/utils/installArgoCD.sh
source kubernetes/bin/utils/installDgraph.sh
source kubernetes/bin/utils/installPostgres.sh
source kubernetes/bin/utils/installWestClusterApp.sh
source kubernetes/bin/utils/secureOPALServer.sh
source kubernetes/bin/utils/secureConfigLoader.sh

if [ $is_debug = false ]; then
  sleep 120

  echo "# East cluster"
  kubectl config use-context k3d-east
  echo "=================================================="

  source kubernetes/bin/utils/installEastClusterApp.sh
  # source kubernetes/bin/utils/verifyLinkerdMulticluster.sh

  echo "# West cluster"
  kubectl config use-context k3d-west
  echo "=================================================="

  # source kubernetes/bin/utils/InstallHMStreaming.sh
fi
