terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/iam_v1_group
resource "nebius_iam_v1_group" "main" {
  parent_id = var.parent_id
  name      = var.group_name
  labels    = var.labels
}
