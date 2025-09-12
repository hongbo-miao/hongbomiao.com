terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
  }
}

# https://docs.nebius.com/terraform-provider/reference/resources/iam_v1_group_membership
resource "nebius_iam_v1_group_membership" "main" {
  parent_id = var.parent_id
  member_id = var.member_id
  labels    = var.labels
}
