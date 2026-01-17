terraform {
  required_providers {
    argocd = {
      source = "argoproj-labs/argocd"
    }
  }
}

# https://registry.terraform.io/providers/argoproj-labs/argocd/latest/docs/resources/project
resource "argocd_project" "main" {
  metadata {
    name      = var.project_name
    namespace = var.argo_cd_namespace
  }
  spec {
    source_repos = ["*"]
    destination {
      server    = "https://kubernetes.default.svc"
      namespace = "*"
    }
    dynamic "namespace_resource_whitelist" {
      for_each = var.namespace_resource_whitelist
      content {
        group = namespace_resource_whitelist.value.group
        kind  = namespace_resource_whitelist.value.kind
      }
    }
    dynamic "cluster_resource_whitelist" {
      for_each = var.cluster_resource_whitelist
      content {
        group = cluster_resource_whitelist.value.group
        kind  = cluster_resource_whitelist.value.kind
      }
    }
  }
}
