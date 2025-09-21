#!/usr/bin/env bash
set -e

echo "# Setup Multipass"
# https://andreipope.github.io/tutorials/create-a-cluster-with-multipass-and-k3s
multipass launch --name=west-master --cpus=8 --memory=56g --disk=512g
# https://www.rancher.co.jp/docs/k3s/latest/en/installation/
multipass exec west-master -- \
  bash -c 'curl --silent --fail --show-error --location https://get.k3s.io | INSTALL_K3S_EXEC="server --disable traefik" K3S_KUBECONFIG_MODE="644" sh -'
multipass mount "${PWD}/data" west-master:/data

# multipass info west-master
# multipass exec west-master -- bash

# multipass delete west-master
# multipass purge
echo "=================================================="

echo "# Get kubeconfig"
multipass exec west-master -- \
  cat /etc/rancher/k3s/k3s.yaml > k3s-west-master-kubeconfig.yaml

# apiVersion: v1
# clusters:
# - cluster:
#     certificate-authority-data: xxx
#     server: https://192.168.205.3:6443
#   name: west-master
# contexts:
# - context:
#     cluster: west-master
#     user: admin
#   name: admin@west-master
# current-context: admin@west-master
# kind: Config
# preferences: {}
# users:
# - name: admin
#   user:
#     client-certificate-data: xxx
#     client-key-data: xxx

WEST_MASTER_IP=$(multipass info west-master | grep IPv4 | awk '{print $2}')
perl -pi -e "s/127.0.0.1/${WEST_MASTER_IP}/" k3s-west-master-kubeconfig.yaml
perl -p0i -e "s/name: default/name: west-master/" k3s-west-master-kubeconfig.yaml
perl -p0i -e "s/cluster: default/cluster: west-master/" k3s-west-master-kubeconfig.yaml
perl -p0i -e "s/user: default/user: admin/" k3s-west-master-kubeconfig.yaml
perl -p0i -e "s/name: default/name: admin\@west-master/" k3s-west-master-kubeconfig.yaml
perl -p0i -e "s/current-context: default/current-context: admin\@west-master/" k3s-west-master-kubeconfig.yaml
perl -p0i -e "s/name: default/name: admin/" k3s-west-master-kubeconfig.yaml
echo "=================================================="

echo "# Export kubeconfig"
export KUBECONFIG="${PWD}/k3s-west-master-kubeconfig.yaml"

# Note this will replace existing kubeconfig
# cp k3s-west-master-kubeconfig.yaml ~/.kube/config
echo "=================================================="

echo "# Install Ingress"
helm upgrade \
  ingress-nginx \
  ingress-nginx \
  --install \
  --repo=https://kubernetes.github.io/ingress-nginx \
  --namespace=ingress-nginx \
  --create-namespace
echo "=================================================="
