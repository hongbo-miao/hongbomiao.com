terraform {
  required_providers {
    grafana = {
      source = "registry.opentofu.org/grafana/grafana"
    }
  }
}

# https://search.opentofu.org/provider/grafana/grafana/latest/docs/resources/folder
resource "grafana_folder" "main" {
  org_id = var.org_id
  title  = var.title
}
