terraform {
  required_providers {
    airbyte = {
      source = "registry.opentofu.org/airbytehq/airbyte"
    }
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/data-sources/connector_configuration
data "airbyte_connector_configuration" "main" {
  connector_name = "destination-snowflake"
  configuration = {
    host                = var.snowflake_host
    warehouse           = var.snowflake_warehouse
    database            = var.snowflake_database
    schema              = var.snowflake_schema
    role                = var.snowflake_role
    username            = var.snowflake_user_name
    disable_type_dedupe = false
  }
  configuration_secrets = {
    credentials = {
      key_pair_authentication = {
        private_key          = var.snowflake_user_private_key
        private_key_password = var.snowflake_user_private_key_passphrase
      }
    }
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/resources/destination
resource "airbyte_destination" "main" {
  name          = var.name
  workspace_id  = var.workspace_id
  definition_id = data.airbyte_connector_configuration.main.definition_id
  configuration = data.airbyte_connector_configuration.main.configuration_json
}
