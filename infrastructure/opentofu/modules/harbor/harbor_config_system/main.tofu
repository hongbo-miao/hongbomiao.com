terraform {
  required_providers {
    harbor = {
      source = "registry.opentofu.org/goharbor/harbor"
    }
  }
}

# https://search.opentofu.org/provider/goharbor/harbor/latest/docs/resources/config_system
resource "harbor_config_system" "main" {
  project_creation_restriction = var.project_creation_restriction
}
