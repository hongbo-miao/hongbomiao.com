terraform {
  required_providers {
    harbor = {
      source = "goharbor/harbor"
    }
  }
}

# https://registry.terraform.io/providers/goharbor/harbor/latest/docs/resources/robot_account
resource "harbor_robot_account" "main" {
  name   = var.name
  secret = var.secret
  level  = "system"
  dynamic "permissions" {
    for_each = var.project_names
    content {
      access {
        action   = "pull"
        resource = "repository"
      }
      kind      = "project"
      namespace = permissions.value
    }
  }
}
