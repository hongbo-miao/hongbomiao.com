terraform {
  required_providers {
    argocd = {
      source = "registry.opentofu.org/argoproj-labs/argocd"
    }
  }
}

# https://search.opentofu.org/provider/argoproj-labs/argocd/latest/docs/resources/project
resource "argocd_project" "main" {
  metadata {
    name      = var.project_name
    namespace = var.argo_cd_namespace
  }
  spec {
    source_repos = ["*"]
    dynamic "destination" {
      for_each = var.destination_namespaces
      content {
        server    = "https://kubernetes.default.svc"
        namespace = destination.value
      }
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
