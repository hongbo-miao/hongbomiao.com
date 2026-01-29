terraform {
  required_providers {
    grafana = {
      source = "grafana/grafana"
    }
  }
}

# https://registry.terraform.io/providers/grafana/grafana/latest/docs/resources/folder
resource "grafana_folder" "main" {
  org_id = var.org_id
  title  = var.title
}
