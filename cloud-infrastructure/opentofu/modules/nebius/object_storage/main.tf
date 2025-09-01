terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/storage_v1_bucket
resource "nebius_storage_v1_bucket" "main" {
  parent_id             = var.project_id
  name                  = var.object_storage_name
  default_storage_class = var.default_storage_class
  versioning_policy     = var.versioning_policy
  labels                = var.labels
}
