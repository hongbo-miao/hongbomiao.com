terraform {
  required_providers {
    airbyte = {
      source = "registry.opentofu.org/airbytehq/airbyte"
    }
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/resources/source_file
resource "airbyte_source_file" "main" {
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
