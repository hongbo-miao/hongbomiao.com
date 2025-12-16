terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/vpc_v1_network
resource "nebius_vpc_v1_network" "main" {
  parent_id = var.project_id
  name      = var.network_name
  labels    = var.labels

  ipv4_private_pools = {
    pools = [
      {
        id = var.ipv4_private_pool_id
      }
    ]
  }
}
