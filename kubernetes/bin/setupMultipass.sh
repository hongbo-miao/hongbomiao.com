#!/usr/bin/env bash
set -e

echo "# Setup Multipass"
# https://andreipope.github.io/tutorials/create-a-cluster-with-multipass-and-k3s
multipass launch --name=west-master --cpus=6 --mem=24g --disk=256g
multipass exec west-master -- \
  bash -c 'curl -sfL https://get.k3s.io | K3S_KUBECONFIG_MODE="644" sh -'
multipass mount "${PWD}/data" west-master:/data
multipass exec west-master -- \
  cat /etc/rancher/k3s/k3s.yaml > west-master-k3s.yaml
echo "=================================================="

echo "# Update kubeconfig"
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
perl -pi -e "s/127.0.0.1/${WEST_MASTER_IP}/" west-master-k3s.yaml
perl -p0i -e "s/name: default/name: west-master/" west-master-k3s.yaml
perl -p0i -e "s/cluster: default/cluster: west-master/" west-master-k3s.yaml
perl -p0i -e "s/user: default/user: admin/" west-master-k3s.yaml
perl -p0i -e "s/name: default/name: admin\@west-master/" west-master-k3s.yaml
perl -p0i -e "s/current-context: default/current-context: admin\@west-master/" west-master-k3s.yaml
perl -p0i -e "s/name: default/name: admin/" west-master-k3s.yaml
echo "=================================================="

echo "# Export kubeconfig"
export KUBECONFIG=${PWD}/west-master-k3s.yaml

# multipass info west-master
# multipass exec west-master -- bash

# multipass delete west-master
# multipass purge
echo "=================================================="
