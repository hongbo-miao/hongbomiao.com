terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/harbor/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.15.0"
    }
    # https://registry.terraform.io/providers/goharbor/harbor/latest
    harbor = {
      source  = "goharbor/harbor"
      version = "3.11.2"
    }
  }
  required_version = ">= 1.10"
}
