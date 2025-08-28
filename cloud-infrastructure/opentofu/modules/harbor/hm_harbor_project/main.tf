terraform {
  required_providers {
    harbor = {
      source = "goharbor/harbor"
    }
  }
}

# https://registry.terraform.io/providers/goharbor/harbor/latest/docs/resources/project
resource "harbor_project" "main" {
  name                   = var.name
  public                 = var.public
  registry_id            = var.registry_id
  auto_sbom_generation   = true
  vulnerability_scanning = true
}
