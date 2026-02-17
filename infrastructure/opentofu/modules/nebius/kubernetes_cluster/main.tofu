terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/mk8s_v1_cluster
resource "nebius_mk8s_v1_cluster" "main" {
  parent_id = var.project_id
  name      = var.kubernetes_cluster_name
  control_plane = {
    subnet_id         = var.vpc_subnet_id
    etcd_cluster_size = 3
    version           = var.kubernetes_version
    endpoints = {
      public_endpoint = {}
    }
  }
  kube_network = {
    service_cidrs = var.service_cidrs
  }
  labels = var.labels
}
