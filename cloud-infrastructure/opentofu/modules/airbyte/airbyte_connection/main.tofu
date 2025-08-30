terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/connection
resource "airbyte_connection" "main" {
  name                                 = "${var.destination_name}-connection"
  source_id                            = var.source_id
  destination_id                       = var.destination_id
  non_breaking_schema_updates_behavior = var.non_breaking_schema_updates_behavior
  status                               = var.status
  schedule = {
    schedule_type   = var.schedule_type
    cron_expression = var.schedule_cron_expression
  }
  configurations = {
    streams = var.streams
  }
}
