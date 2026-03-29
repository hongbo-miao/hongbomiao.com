terraform {
  required_providers {
    airbyte = {
      source = "registry.opentofu.org/airbytehq/airbyte"
    }
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/data-sources/connector_configuration
data "airbyte_connector_configuration" "main" {
  connector_name = "source-file"
  configuration = {
    dataset_name = var.dataset_name
    format       = "csv"
    url          = var.url
    provider = {
      https_public_web = {}
    }
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/resources/source
resource "airbyte_source" "main" {
  name          = var.name
  workspace_id  = var.workspace_id
  definition_id = data.airbyte_connector_configuration.main.definition_id
  configuration = data.airbyte_connector_configuration.main.configuration_json
}
