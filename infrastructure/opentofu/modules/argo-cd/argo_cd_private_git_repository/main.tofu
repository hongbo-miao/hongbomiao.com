terraform {
  required_providers {
    argocd = {
      source = "registry.opentofu.org/argoproj-labs/argocd"
    }
  }
}

# https://search.opentofu.org/provider/argoproj-labs/argocd/latest/docs/resources/repository
resource "argocd_repository" "main" {
  repo            = var.repository
  username        = "git"
  ssh_private_key = var.ssh_private_key
  insecure        = false
}
