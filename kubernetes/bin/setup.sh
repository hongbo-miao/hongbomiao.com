#!/usr/bin/env bash
set -e


is_debug=false
# source kubernetes/bin/utils/focusOPADebug.sh

export ORG_DOMAIN="k8s-hongbomiao.com"

if [ $is_debug = false ]; then
  CLUSTERS=("west" "east")
else
  CLUSTERS=("west")
fi

source kubernetes/bin/utils/setupK3d.sh "${CLUSTERS[@]}"

if [ $is_debug = false ]; then
  source kubernetes/bin/utils/installLinkerd.sh "${CLUSTERS[@]}"
fi

source kubernetes/bin/utils/installIngress.sh "${CLUSTERS[@]}"
if [ $is_debug = false ]; then
  source kubernetes/bin/utils/patchIngress.sh "${CLUSTERS[@]}"
fi
sleep 30

if [ $is_debug = false ]; then
  source kubernetes/bin/utils/installLinkerdMulticluster.sh "${CLUSTERS[@]}"
fi


echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="


if [ $is_debug = false ]; then
  source kubernetes/bin/utils/installLinkerdViz.sh
fi

if [ $is_debug = false ]; then
  echo "# List gateways"
  linkerd multicluster gateways
  echo "=================================================="
fi

if [ $is_debug = false ]; then
  source kubernetes/bin/utils/installLinkerdJaeger.sh
  # source kubernetes/bin/utils/installLinkerdBuoyant.sh

  source kubernetes/bin/utils/installFluentBit.sh
  source kubernetes/bin/utils/installMinIO.sh
  # source kubernetes/bin/utils/installYugabyte.sh
  source kubernetes/bin/utils/installKafka.sh
  source kubernetes/bin/utils/installDebezium.sh
fi

source kubernetes/bin/utils/installElastic.sh
source kubernetes/bin/utils/installArgoCD.sh

source kubernetes/bin/utils/installPostgres.sh
source kubernetes/bin/utils/migratePostgres.sh
source kubernetes/bin/utils/installWestClusterApp.sh
source kubernetes/bin/utils/secureOPALServer.sh
source kubernetes/bin/utils/secureConfigServer.sh

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
