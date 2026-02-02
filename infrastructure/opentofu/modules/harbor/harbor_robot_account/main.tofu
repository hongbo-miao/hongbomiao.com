terraform {
  required_providers {
    harbor = {
      source = "registry.opentofu.org/goharbor/harbor"
    }
  }
}

# https://search.opentofu.org/provider/goharbor/harbor/latest/docs/resources/robot_account
resource "harbor_robot_account" "main" {
  name   = var.name
  secret = var.secret
  level  = "system"
  dynamic "permissions" {
    for_each = var.project_names
    content {
      dynamic "access" {
        for_each = var.actions
        content {
          action   = access.value
          resource = "repository"
        }
      }
      kind      = "project"
      namespace = permissions.value
    }
  }
}
