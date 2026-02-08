terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/aws/kubernetes/terraform.tfstate"
  }
  required_providers {
    # https://search.opentofu.org/provider/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.31.0"
    }
    # https://search.opentofu.org/provider/hashicorp/external/latest
    external = {
      source  = "hashicorp/external"
      version = "2.3.5"
    }
    # https://search.opentofu.org/provider/hashicorp/helm/latest
    helm = {
      source  = "hashicorp/helm"
      version = "3.1.1"
    }
    # https://search.opentofu.org/provider/gavinbunney/kubectl/latest
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "1.19.0"
    }
    # https://search.opentofu.org/provider/hashicorp/kubernetes/latest
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "3.0.1"
    }
  }
  required_version = ">= 1.11"
}
