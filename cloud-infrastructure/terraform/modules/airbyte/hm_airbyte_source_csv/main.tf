terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/source_file
resource "airbyte_source_file" "hm_airbyte_source_csv" {
  name         = var.name
  workspace_id = var.workspace_id
  configuration = {
    dataset_name = var.dataset_name
    format       = "csv"
    provider = {
      https_public_web = {}
    }
    url = var.url
  }
}
