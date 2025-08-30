terraform {
  required_providers {
    harbor = {
      source = "goharbor/harbor"
    }
  }
}

# https://registry.terraform.io/providers/goharbor/harbor/latest/docs/resources/garbage_collection
resource "harbor_garbage_collection" "main" {
  schedule        = var.schedule
  delete_untagged = true
  workers         = var.worker_number
}
