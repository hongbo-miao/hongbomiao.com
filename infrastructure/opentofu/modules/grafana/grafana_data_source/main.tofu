terraform {
  required_providers {
    grafana = {
      source = "grafana/grafana"
    }
  }
}

# https://registry.terraform.io/providers/grafana/grafana/latest/docs/resources/data_source
resource "grafana_data_source" "main" {
  org_id                   = var.org_id
  name                     = var.name
  type                     = var.type
  url                      = var.url
  is_default               = var.is_default
  http_headers             = var.http_headers
  json_data_encoded        = var.json_data_encoded
  secure_json_data_encoded = var.secure_json_data_encoded
}
