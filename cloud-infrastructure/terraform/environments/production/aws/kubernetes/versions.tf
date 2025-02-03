terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/data/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/gavinbunney/kubectl/latest
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "1.19.0"
    }
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.85.0"
    }
    # https://registry.terraform.io/providers/hashicorp/external/latest
    external = {
      source  = "hashicorp/external"
      version = "2.3.4"
    }
    # https://registry.terraform.io/providers/hashicorp/helm/latest
    helm = {
      source  = "hashicorp/helm"
      version = "2.17.0"
    }
    # https://registry.terraform.io/providers/hashicorp/kubernetes/latest
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "2.35.1"
    }
  }
  required_version = ">= 1.8"
}
