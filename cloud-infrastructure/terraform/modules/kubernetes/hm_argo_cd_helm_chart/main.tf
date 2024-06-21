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
  values     = [file(var.my_values_yaml_path)]
  wait       = true
  namespace  = var.namespace
}
