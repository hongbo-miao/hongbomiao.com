terraform {
  required_providers {
    zitadel = {
      source = "registry.opentofu.org/zitadel/zitadel"
    }
  }
}

# https://search.opentofu.org/provider/zitadel/zitadel/latest/docs/resources/project_role
resource "zitadel_project_role" "main" {
  org_id       = var.org_id
  project_id   = var.project_id
  role_key     = var.role_key
  display_name = var.display_name
}
