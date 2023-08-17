terraform {
  backend "remote" {
    organization = "hongbomiao"
    workspaces {
      name = "hm-workspace"
    }
  }
  required_providers {
    # https://registry.terraform.io/providers/hashicorp/aws/latest
    aws = {
      source  = "hashicorp/aws"
      version = "5.12.0"
    }
  }
  # https://app.terraform.io/app/hongbomiao/workspaces/hm-workspace/settings/general
  required_version = ">= 1.5"
}

provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "hm_ec2_instance" {
  ami           = "ami-08d70e59c07c61a3a"
  instance_type = "t2.nano"
  tags = {
    Name = var.instance_name
  }
}
