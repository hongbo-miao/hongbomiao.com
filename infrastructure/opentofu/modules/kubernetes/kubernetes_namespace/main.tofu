terraform {
  required_providers {
    kubernetes = {
      source = "registry.opentofu.org/hashicorp/kubernetes"
    }
  }
}

# https://search.opentofu.org/provider/hashicorp/kubernetes/latest/docs/resources/namespace_v1
resource "kubernetes_namespace_v1" "main" {
  metadata {
    name   = var.kubernetes_namespace
    labels = var.labels
  }
}
