terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/destination_snowflake
resource "airbyte_destination_snowflake" "hm_airbyte_destination_snowflake" {
  name         = var.name
  workspace_id = var.workspace_id
  configuration = {
    host      = var.snowflake_host
    warehouse = var.snowflake_warehouse
    database  = var.snowflake_database
    schema    = var.snowflake_schema
    role      = var.snowflake_role
    username  = var.snowflake_user_name
    credentials = {
      key_pair_authentication = {
        private_key          = var.snowflake_user_private_key
        private_key_password = var.snowflake_user_private_key_passphrase
      }
    }
    retention_period_days = var.data_retention_days
    raw_data_schema       = "_AIRBYTE_INTERNAL"
    disable_type_dedupe   = false
  }
}
