terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/aws/general/terraform.tfstate"
  }
  required_providers {
    # https://search.opentofu.org/provider/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.32.0"
    }
    # https://search.opentofu.org/provider/hashicorp/awscc/latest
    awscc = {
      source  = "hashicorp/awscc"
      version = "1.71.0"
    }
  }
  required_version = ">= 1.11"
}
