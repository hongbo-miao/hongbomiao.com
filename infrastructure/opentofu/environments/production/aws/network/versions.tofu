terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "production-hm-opentofu-bucket"
    key    = "production/aws/network/terraform.tfstate"
  }
  required_providers {
    # https://search.opentofu.org/provider/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "6.30.0"
    }
  }
  required_version = ">= 1.11"
}
