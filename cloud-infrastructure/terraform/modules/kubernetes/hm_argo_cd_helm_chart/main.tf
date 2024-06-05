terraform {
  required_providers {
    helm = {
      source = "hashicorp/helm"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/helm/latest/docs/resources/release
resource "helm_release" "hm_argo_cd_helm_chart" {
  repository = "https://argoproj.github.io/argo-helm"
  chart      = "argo-cd"
  name       = var.name
  version    = var.argo_cd_version
  namespace  = var.namespace
}
