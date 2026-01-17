terraform {
  required_providers {
    argocd = {
      source = "argoproj-labs/argocd"
    }
  }
}

# https://registry.terraform.io/providers/claranet/argocd/latest/docs/resources/application
resource "argocd_application" "main" {
  metadata {
    name      = var.application_name
    namespace = var.argo_cd_namespace
  }

  spec {
    project = var.argo_cd_project
    source {
      repo_url        = var.repository
      target_revision = "main"
      path            = var.path
      directory {
        recurse = true
        include = var.include
      }
    }
    destination {
      server    = "https://kubernetes.default.svc"
      namespace = var.argo_cd_namespace
    }
    sync_policy {
      automated {
        prune     = true
        self_heal = true
      }
    }
  }
}
