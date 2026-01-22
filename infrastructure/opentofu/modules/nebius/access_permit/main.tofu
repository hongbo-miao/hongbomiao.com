terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/iam_v1_access_permit
resource "nebius_iam_v1_access_permit" "main" {
  parent_id   = var.parent_id
  resource_id = var.resource_id
  # https://docs.nebius.com/iam/authorization/roles
  role   = var.role
  labels = var.labels
}
