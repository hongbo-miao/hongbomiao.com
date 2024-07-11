terraform {
  required_providers {
    helm = {
      source = "hashicorp/helm"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/helm/latest/docs/resources/release
resource "helm_release" "hm_helm_chart" {
  repository   = var.repository
  chart        = var.chart_name
  version      = var.chart_version
  name         = var.release_name
  namespace    = var.namespace
  values       = var.my_values_yaml_path != "" ? [file(var.my_values_yaml_path)] : []
  reset_values = true
  wait         = true
}
