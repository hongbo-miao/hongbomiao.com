terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "production/airbyte/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/airbytehq/airbyte/latest
    airbyte = {
      source  = "airbytehq/airbyte"
      version = "0.13.0"
    }
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.7.0"
    }
  }
  # terraform version
  required_version = ">= 1.7"
}
