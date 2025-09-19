terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/iam_v1_service_account
resource "nebius_iam_v1_service_account" "main" {
  parent_id = var.project_id
  name      = var.service_account_name
  labels    = var.labels
}
