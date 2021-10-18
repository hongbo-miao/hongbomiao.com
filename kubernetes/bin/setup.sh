#!/usr/bin/env bash

set -e


export ORG_DOMAIN="k8s-hongbomiao.com"
CLUSTERS=("west" "east")

source kubernetes/bin/utils/setupK3d.sh "${CLUSTERS[@]}"
source kubernetes/bin/utils/installLinkerd.sh "${CLUSTERS[@]}"

source kubernetes/bin/utils/installIngress.sh "${CLUSTERS[@]}"
source kubernetes/bin/utils/patchIngress.sh "${CLUSTERS[@]}"
sleep 30

source kubernetes/bin/utils/installLinkerdMulticluster.sh "${CLUSTERS[@]}"


# West cluster
echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="


source kubernetes/bin/utils/installLinkerdViz.sh

# List gateways
echo "# List gateways"
linkerd multicluster gateways
echo "=================================================="


source kubernetes/bin/utils/installLinkerdJaeger.sh
# source kubernetes/bin/utils/installLinkerdBuoyant.sh

source kubernetes/bin/utils/installElastic.sh
source kubernetes/bin/utils/installFluentBit.sh
source kubernetes/bin/utils/installMinIO.sh
source kubernetes/bin/utils/installYugabyte.sh
source kubernetes/bin/utils/installKafka.sh
source kubernetes/bin/utils/installArgoCD.sh

source kubernetes/bin/utils/installWestClusterApp.sh
source kubernetes/bin/utils/secureOPALServer.sh
sleep 120


# East cluster
echo "# East cluster"
kubectl config use-context k3d-east
echo "=================================================="


source kubernetes/bin/utils/installEastClusterApp.sh


# source kubernetes/bin/utils/verifyLinkerdMulticluster.sh


# West cluster
echo "# West cluster"
kubectl config use-context k3d-west
echo "=================================================="

# source kubernetes/bin/utils/InstallHMStreaming.sh
