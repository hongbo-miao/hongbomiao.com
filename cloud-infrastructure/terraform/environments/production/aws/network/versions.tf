terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/network/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.84.0"
    }
  }
  required_version = ">= 1.8"
}
