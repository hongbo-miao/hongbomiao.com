#!/usr/bin/env bash
set -e

# https://andreipope.github.io/tutorials/create-a-cluster-with-multipass-and-k3s
multipass launch --name=west-master --cpus=6 --mem=16g --disk=128g
multipass exec west-master -- \
  bash -c 'curl -sfL https://get.k3s.io | K3S_KUBECONFIG_MODE="644" sh -'
multipass mount $HOME/Clouds/Git/hongbomiao.com/kubernetes/data west-master:/data

multipass exec west-master -- \
  cat /etc/rancher/k3s/k3s.yaml > west-master-k3s.yaml

WEST_MASTER_IP=$(multipass info west-master | grep IPv4 | awk '{print $2}')
sed -i '' "s/127.0.0.1/${WEST_MASTER_IP}/" west-master-k3s.yaml
export KUBECONFIG=${PWD}/west-master-k3s.yaml

# multipass info west-master
# multipass exec west-master -- bash

# multipass delete west-master
# multipass purge
