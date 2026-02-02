terraform {
  required_providers {
    grafana = {
      source = "registry.opentofu.org/grafana/grafana"
    }
  }
}

# https://search.opentofu.org/provider/grafana/grafana/latest/docs/resources/organization
resource "grafana_organization" "main" {
  name = var.name
  lifecycle {
    ignore_changes = [
      admin_user,
      admins,
      create_users,
      editors,
      users_without_access,
      viewers,
    ]
  }
}
