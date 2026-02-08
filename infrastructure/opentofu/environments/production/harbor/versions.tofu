terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/harbor/terraform.tfstate"
  }
  required_providers {
    # https://search.opentofu.org/provider/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.31.0"
    }
    # https://search.opentofu.org/provider/goharbor/harbor/latest
    harbor = {
      source  = "goharbor/harbor"
      version = "3.11.3"
    }
  }
  required_version = ">= 1.11"
}
