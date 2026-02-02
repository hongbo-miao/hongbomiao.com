terraform {
  required_providers {
    harbor = {
      source = "registry.opentofu.org/goharbor/harbor"
    }
  }
}

# https://search.opentofu.org/provider/goharbor/harbor/latest/docs/resources/project
resource "harbor_project" "main" {
  name                   = var.name
  public                 = var.public
  registry_id            = var.registry_id
  auto_sbom_generation   = true
  vulnerability_scanning = true
}
