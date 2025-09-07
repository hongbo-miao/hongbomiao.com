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
    namespace_resource_whitelist {
      group = "*"
      kind  = "*"
    }
    cluster_resource_whitelist {
      group = ""
      kind  = "Namespace"
    }
    cluster_resource_whitelist {
      group = "apiextensions.k8s.io"
      kind  = "CustomResourceDefinition"
    }
    cluster_resource_whitelist {
      group = "networking.k8s.io"
      kind  = "IngressClass"
    }
    cluster_resource_whitelist {
      group = "rbac.authorization.k8s.io"
      kind  = "ClusterRole"
    }
    cluster_resource_whitelist {
      group = "rbac.authorization.k8s.io"
      kind  = "ClusterRoleBinding"
    }
  }
}
