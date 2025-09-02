terraform {
  required_providers {
    argocd = {
      source = "argoproj-labs/argocd"
    }
  }
}

# https://registry.terraform.io/providers/argoproj-labs/argocd/latest/docs/resources/repository
resource "argocd_repository" "main" {
  repo            = var.repository
  username        = "git"
  ssh_private_key = var.ssh_private_key
  insecure        = false
}
