terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/mk8s_v1_node_group
resource "nebius_mk8s_v1_node_group" "main" {
  parent_id = var.parent_id
  name      = var.name
  autoscaling = {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }
  template = {
    resources = {
      platform = var.platform
      preset   = var.preset
    }
    os = var.os
  }
  labels = var.labels
}
