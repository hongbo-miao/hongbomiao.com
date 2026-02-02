terraform {
  required_providers {
    harbor = {
      source = "registry.opentofu.org/goharbor/harbor"
    }
  }
}

# https://search.opentofu.org/provider/goharbor/harbor/latest/docs/resources/registry
resource "harbor_registry" "main" {
  name          = var.name
  provider_name = var.provider_name
  endpoint_url  = var.endpoint_url
}
