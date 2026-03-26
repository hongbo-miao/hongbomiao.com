terraform {
  required_providers {
    airbyte = {
      source = "registry.opentofu.org/airbytehq/airbyte"
    }
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/data-sources/connector_configuration
data "airbyte_connector_configuration" "main" {
  connector_name = "source-mssql"
  configuration = {
    host     = var.microsoft_sql_server_host
    port     = var.microsoft_sql_server_port
    database = var.microsoft_sql_server_database
    schemas  = [var.microsoft_sql_server_schema]
    replication_method = {
      read_changes_using_change_data_capture_cdc = {
        initial_load_timeout_hours           = 8
        initial_waiting_seconds              = var.initial_waiting_time_s
        queue_size                           = 10000
        invalid_cdc_cursor_position_behavior = "Re-sync data"
      }
    }
    ssl_method = {
      encrypted_trust_server_certificate = {}
    }
  }
  configuration_secrets = {
    username = var.microsoft_sql_server_user_name
    password = var.microsoft_sql_server_password
  }
}

# https://search.opentofu.org/provider/airbytehq/airbyte/latest/docs/resources/source
resource "airbyte_source" "main" {
  name          = var.name
  workspace_id  = var.workspace_id
  definition_id = data.airbyte_connector_configuration.main.definition_id
  configuration = data.airbyte_connector_configuration.main.configuration_json
}
