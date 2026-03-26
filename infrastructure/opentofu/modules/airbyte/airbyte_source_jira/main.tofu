terraform {
  required_providers {
    airbyte = {
      source = "registry.opentofu.org/airbytehq/airbyte"
    }
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/data-sources/connector_configuration
data "airbyte_connector_configuration" "main" {
  connector_name = "source-jira"
  configuration = {
    domain = var.jira_domain
    email  = var.jira_user_email
  }
  configuration_secrets = {
    api_token = var.jira_user_api_token
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/resources/source
resource "airbyte_source" "main" {
  name          = var.name
  workspace_id  = var.workspace_id
  definition_id = data.airbyte_connector_configuration.main.definition_id
  configuration = data.airbyte_connector_configuration.main.configuration_json
}
