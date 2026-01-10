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
  ipv4_private_pools = var.should_use_private_network_pools ? {
    use_network_pools = true
  } : null
  ipv4_public_pools = var.should_use_public_network_pools ? {
    use_network_pools = true
  } : null
  labels = var.labels
}
