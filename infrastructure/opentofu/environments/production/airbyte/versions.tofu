terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/airbyte/terraform.tfstate"
  }
  required_providers {
    # https://search.opentofu.org/provider/airbytehq/airbyte/latest
    airbyte = {
      source  = "airbytehq/airbyte"
      version = "1.1.0"
    }
    # https://search.opentofu.org/provider/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.39.0"
    }
  }
  required_version = ">= 1.11"
}
