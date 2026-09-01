terraform {
  # Local backend: this demo's Ranger policy state is throwaway kind-cluster state, not the
  # production S3 backend used by infrastructure/opentofu/environments/production/*.
  required_providers {
    # https://registry.terraform.io/providers/gr8-toolkit/ranger/latest
    ranger = {
      source  = "registry.terraform.io/gr8-toolkit/ranger"
      version = "0.1.0"
    }
  }
  required_version = ">= 1.11"
}
