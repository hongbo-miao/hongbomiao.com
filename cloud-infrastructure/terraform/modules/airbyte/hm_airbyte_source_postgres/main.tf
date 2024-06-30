terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/source_postgres
resource "airbyte_source_postgres" "hm_airbyte_source_postgres" {
  name         = var.name
  workspace_id = var.workspace_id
  configuration = {
    host     = var.postgres_host
    port     = var.postgres_port
    username = var.postgres_user_name
    password = var.postgres_password
    database = var.postgres_database
    schemas  = var.postgres_schemas
    replication_method = {
      read_changes_using_write_ahead_log_cdc = {
        replication_slot = var.postgres_replication_slot
        publication      = var.postgres_publication
      }
    }
    ssl_mode = {
      require = {}
    }
  }
}
