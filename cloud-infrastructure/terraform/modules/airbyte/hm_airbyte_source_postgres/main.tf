terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/source_postgres
resource "airbyte_source_postgres" "main" {
  name         = var.name
  workspace_id = var.workspace_id
  configuration = {
    host     = var.postgres_host
    port     = var.postgres_port
    username = var.postgres_user_name
    password = var.postgres_password
    database = var.postgres_database
    schemas  = [var.postgres_schema]
    replication_method = {
      read_changes_using_write_ahead_log_cdc = {
        replication_slot                     = "airbyte_${var.postgres_schema}_logical_replication_slot"
        publication                          = "airbyte_${var.postgres_schema}_publication"
        initial_load_timeout_hours           = 8
        initial_waiting_seconds              = var.initial_waiting_time_s
        queue_size                           = 10000
        heartbeat_action_query               = "insert into ${var.postgres_schema}._airbyte_heartbeat (id, timestamp) values (1, now()) on conflict (id) do update set timestamp = excluded.timestamp;"
        lsn_commit_behaviour                 = "After loading Data in the destination"
        invalid_cdc_cursor_position_behavior = "Re-sync data"
      }
    }
    ssl_mode = {
      require = {}
    }
    tunnel_method = var.tunnel_method
  }
}
