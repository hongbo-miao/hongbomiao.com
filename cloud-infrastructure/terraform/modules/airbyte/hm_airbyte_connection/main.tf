terraform {
  required_providers {
    airbyte = {
      source = "airbytehq/airbyte"
    }
  }
}

# https://registry.terraform.io/providers/airbytehq/airbyte/latest/docs/resources/connection
resource "airbyte_connection" "hm_airbyte_connection" {
  name           = var.name
  source_id      = var.source_id
  destination_id = var.destination_id
  configurations = {
    streams = var.streams
  }
}
