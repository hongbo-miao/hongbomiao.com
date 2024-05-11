terraform {
  backend "s3" {
    region = "us-west-2"
    bucket = "hm-terraform-bucket"
    key    = "development/aws/network/terraform.tfstate"
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.49.0"
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

# Amazon VPC
data "aws_vpc" "hm_amazon_vpc" {
  default = true
}
data "aws_subnets" "hm_amazon_vpc_subnets" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.hm_amazon_vpc.id]
  }
}
