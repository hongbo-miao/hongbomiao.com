terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "development/aws/data/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.48.0"
    }
    # https://registry.terraform.io/providers/hashicorp/awscc/latest
    awscc = {
      source  = "hashicorp/awscc"
      version = "0.76.0"
    }
  }
  required_version = ">= 1.7"
}

provider "aws" {
  alias  = "development"
  region = "us-west-2"
}
provider "awscc" {
  alias  = "development"
  region = "us-west-2"
}

# Amazon S3 bucket - hm-development-bucket
module "development_hm_development_bucket_amazon_s3_bucket" {
  providers      = { aws = aws.development }
  source         = "../../../../modules/aws/hm_amazon_s3_bucket"
  s3_bucket_name = "hm-development-bucket"
  environment    = var.environment
  team           = var.team
}
