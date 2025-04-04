terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/aws/general/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.94.1"
    }
    # https://registry.terraform.io/providers/hashicorp/awscc/latest
    awscc = {
      source  = "hashicorp/awscc"
      version = "1.36.0"
    }
  }
  required_version = ">= 1.8"
}
