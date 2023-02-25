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
      version = "4.56.0"
    }
  }
  # https://app.terraform.io/app/hongbomiao/workspaces/hm-workspace/settings/general
  required_version = ">= 1.3"
}

provider "aws" {
  profile = "hm"
  region  = "us-west-2"
}

resource "aws_instance" "hm_cnn_instance" {
  ami           = "ami-08d70e59c07c61a3a"
  instance_type = "t2.nano"
  tags = {
    Name = var.instance_name
  }
}
