terraform {
  required_providers {
    helm = {
      source = "hashicorp/helm"
    }
  }
}

# https://registry.terraform.io/providers/hashicorp/helm/latest/docs/resources/release
resource "helm_release" "main" {
  repository   = var.repository
  chart        = var.chart_name
  version      = var.chart_version
  name         = var.release_name
  namespace    = var.namespace
  values       = var.my_values_yaml_path != "" ? [file(var.my_values_yaml_path)] : var.my_values
  reset_values = true
  wait         = true
}
