terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/vpc_v1_subnet
resource "nebius_vpc_v1_subnet" "main" {
  parent_id  = var.project_id
  network_id = var.network_id
  name       = var.subnet_name
  labels     = var.labels

  ipv4_private_pools = var.private_ipv4_cidr_block != null ? {
    use_network_pools = false
    pools = [
      {
        cidrs = [
          {
            cidr = var.private_ipv4_cidr_block
          }
        ]
      }
    ]
  } : null

  ipv4_public_pools = var.public_ipv4_cidr_block != null ? {
    use_network_pools = false
    pools = [
      {
        cidrs = [
          {
            cidr = var.public_ipv4_cidr_block
          }
        ]
      }
    ]
  } : null
}
