terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/harbor/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.97.0"
    }
    # https://registry.terraform.io/providers/goharbor/harbor/latest
    harbor = {
      source  = "goharbor/harbor"
      version = "3.10.21"
    }
  }
  # terraform version
  required_version = ">= 1.7"
}
