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
      version = "0.6.5"
    }
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.86.1"
    }
  }
  # terraform version
  required_version = ">= 1.7"
}
