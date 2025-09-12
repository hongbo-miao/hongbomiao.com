terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/msp_mlflow_v1alpha1_cluster
resource "nebius_msp_mlflow_v1alpha1_cluster" "main" {
  parent_id           = var.project_id
  network_id          = var.network_id
  name                = var.mlflow_cluster_name
  size                = var.mlflow_cluster_size
  admin_username      = var.admin_username
  admin_password      = var.admin_password
  service_account_id  = var.service_account_id
  public_access       = var.public_access
  storage_bucket_name = var.storage_bucket_name
  labels              = var.labels
}
