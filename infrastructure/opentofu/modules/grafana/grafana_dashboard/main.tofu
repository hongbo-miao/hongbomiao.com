terraform {
  required_providers {
    grafana = {
      source = "grafana/grafana"
    }
  }
}

# https://registry.terraform.io/providers/grafana/grafana/latest/docs/resources/dashboard
locals {
  dashboard_config = jsondecode(var.config_json)
  existing_tags    = try(local.dashboard_config.tags, [])
  combined_tags    = distinct(concat(local.existing_tags, var.tags))
  modified_config = merge(local.dashboard_config, {
    uid  = "${var.uid_prefix}${local.dashboard_config.uid}"
    tags = local.combined_tags
  })
}
resource "grafana_dashboard" "main" {
  org_id      = var.org_id
  folder      = var.folder_uid
  config_json = jsonencode(local.modified_config)
  overwrite   = true
}
