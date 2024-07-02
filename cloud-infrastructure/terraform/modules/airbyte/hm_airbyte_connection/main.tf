terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/connection
resource "airbyte_connection" "hm_airbyte_connection" {
  name           = "${var.destination_name}-connection"
  source_id      = var.source_id
  destination_id = var.destination_id
  configurations = {
    streams = var.streams
  }
  schedule = {
    schedule_type   = var.schedule_type
    cron_expression = var.schedule_cron_expression
  }
  lifecycle {
    ignore_changes = [
      # https://github.com/airbytehq/terraform-provider-airbyte/issues/83
      configurations.streams
    ]
  }
}
