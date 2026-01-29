terraform {
  required_providers {
    grafana = {
      source = "grafana/grafana"
    }
  }
}

# https://registry.terraform.io/providers/grafana/grafana/latest/docs/resources/dashboard
resource "grafana_dashboard" "main" {
  org_id      = var.org_id
  folder      = var.folder_uid
  config_json = var.config_json
}
