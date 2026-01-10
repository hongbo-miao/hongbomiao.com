module "hm_pool" {
  source          = "../../../../modules/nebius/vpc_pool"
  project_id      = var.project_id
  pool_name       = "hm-pool"
  visibility      = "PRIVATE"
  ipv4_cidr_block = "10.10.0.0/15"
  labels          = var.common_labels
}

module "hm_network" {
  source               = "../../../../modules/nebius/vpc_network"
  project_id           = var.project_id
  network_name         = "hm-network"
  ipv4_private_pool_id = module.hm_pool.id
  labels               = var.common_labels
}

module "hm_private_subnet" {
  source                           = "../../../../modules/nebius/vpc_subnet"
  project_id                       = var.project_id
  network_id                       = module.hm_network.id
  subnet_name                      = "hm-private-subnet"
  should_use_private_network_pools = true
  labels                           = var.common_labels
}
