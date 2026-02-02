terraform {
  required_providers {
    harbor = {
      source = "registry.opentofu.org/goharbor/harbor"
    }
  }
}

# https://search.opentofu.org/provider/goharbor/harbor/latest/docs/resources/garbage_collection
resource "harbor_garbage_collection" "main" {
  schedule        = var.schedule
  delete_untagged = true
  workers         = var.worker_number
}
