terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/source_mssql
resource "airbyte_source_mssql" "main" {
  name         = var.name
  workspace_id = var.workspace_id
  configuration = {
    host     = var.microsoft_sql_server_host
    port     = var.microsoft_sql_server_port
    database = var.microsoft_sql_server_database
    username = var.microsoft_sql_server_user_name
    password = var.microsoft_sql_server_password
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
}
