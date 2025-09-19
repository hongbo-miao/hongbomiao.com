terraform {
  backend "s3" {
    bucket  = "production-hm-terraform-bucket"
    key     = "production/nebius/kubernetes/terraform.tfstate"
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
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = "0.5.109"
    }
    # https://registry.terraform.io/providers/hashicorp/helm/latest
    helm = {
      source  = "hashicorp/helm"
      version = "3.0.2"
    }
    # https://registry.terraform.io/providers/gavinbunney/kubectl/latest
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "1.19.0"
    }
    # https://registry.terraform.io/providers/hashicorp/kubernetes/latest
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.38.0"
    }
  }
  required_version = ">= 1.10"
}
