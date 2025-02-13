terraform {
  required_providers {
    harbor = {
      source = "goharbor/harbor"
    }
  }
}

# https://registry.terraform.io/providers/goharbor/harbor/latest/docs/resources/config_system
resource "harbor_config_system" "main" {
  project_creation_restriction = var.project_creation_restriction
}
