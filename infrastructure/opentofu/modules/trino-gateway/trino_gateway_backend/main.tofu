terraform {
  required_providers {
    trinogateway = {
      source = "registry.terraform.io/paragor/trinogateway"
    }
  }
}

# https://registry.terraform.io/providers/paragor/trinogateway/latest/docs/resources/backend
resource "trinogateway_backend" "main" {
  name          = var.name
  proxy_to      = var.proxy_to
  active        = var.active
  routing_group = var.routing_group
  external_url  = var.external_url
}
