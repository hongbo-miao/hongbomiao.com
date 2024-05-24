terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "development/aws/general/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.51.0"
    }
    # https://registry.terraform.io/providers/hashicorp/awscc/latest
    awscc = {
      source  = "hashicorp/awscc"
      version = "0.78.0"
    }
  }
  required_version = ">= 1.8"
}
