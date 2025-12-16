terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/vpc_v1_pool
resource "nebius_vpc_v1_pool" "main" {
  parent_id  = var.project_id
  name       = var.pool_name
  version    = "IPV4"
  visibility = var.visibility
  labels     = var.labels
  cidrs = [
    {
      cidr = var.ipv4_cidr_block
    }
  ]
}
