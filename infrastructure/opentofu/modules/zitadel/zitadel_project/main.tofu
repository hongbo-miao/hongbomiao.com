terraform {
  required_providers {
    zitadel = {
      source = "registry.opentofu.org/zitadel/zitadel"
    }
  }
}

# https://search.opentofu.org/provider/zitadel/zitadel/latest/docs/resources/project
resource "zitadel_project" "main" {
  name                   = var.name
  org_id                 = var.org_id
  project_role_assertion = var.project_role_assertion
  project_role_check     = var.project_role_check
  has_project_check      = var.has_project_check
}
