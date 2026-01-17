terraform {
  required_providers {
    kubernetes = {
      source = "hashicorp/kubernetes"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/kubernetes/latest/docs/resources/namespace_v1
resource "kubernetes_namespace_v1" "main" {
  metadata {
    name   = var.kubernetes_namespace
    labels = var.labels
  }
}
