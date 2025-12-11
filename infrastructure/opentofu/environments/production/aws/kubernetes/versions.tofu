terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/aws/kubernetes/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.26.0"
    }
    # https://registry.terraform.io/providers/hashicorp/external/latest
    external = {
      source  = "hashicorp/external"
      version = "2.3.5"
    }
    # https://registry.terraform.io/providers/hashicorp/helm/latest
    helm = {
      source  = "hashicorp/helm"
      version = "3.1.1"
    }
    # https://registry.terraform.io/providers/gavinbunney/kubectl/latest
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "1.19.0"
    }
    # https://registry.terraform.io/providers/hashicorp/kubernetes/latest
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "3.0.1"
    }
  }
  required_version = ">= 1.10"
}
