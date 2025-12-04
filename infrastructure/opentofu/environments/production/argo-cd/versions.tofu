terraform {
  backend "s3" {
    bucket  = "production-hm-terraform-bucket"
    key     = "production/argo-cd/terraform.tfstate"
    region  = "us-central1"
    profile = "nebius"
    endpoints = {
      "s3" = "https://storage.us-central1.nebius.cloud"
    }
    # Skip AWS-specific validations as Nebius provides different settings
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
    skip_region_validation      = true
  }
  required_providers {
    // https://registry.terraform.io/providers/argoproj-labs/argocd/latest
    argocd = {
      source  = "argoproj-labs/argocd"
      version = "7.12.1"
    }
  }
  required_version = ">= 1.10"
}
